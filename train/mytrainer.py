import logging
import os
import sys
# from apex import amp
# from fairscale.nn.data_parallel import (
#     FullyShardedDataParallel as FullyShardedDDP,
#     ShardedDataParallel as ShardedDDP,
# )
# from fairscale.nn.wrap import auto_wrap
import torch
from torch.nn import functional as F, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import Trainer

from logger import BITDISTILLER_DEBUG, FSDP_DEBUG, log_fsdp_debug
logger = logging.getLogger(__name__)


INT_MAX = 2_147_483_647
def check_for_nan_or_inf(tensor, name=""):
    if int(os.environ.get('LOCAL_RANK', '0')) != 0:
        return
    
    if not torch.is_floating_point(tensor):
        return  # skip int/bool tensors

    if torch.isfinite(tensor).all():
        return

    logger.debug(f"\n[!] NaN or Inf detected in: {name}", file=sys.stderr)
    logger.debug(f"    Shape: {tensor.shape}, Device: {tensor.device}", file=sys.stderr)
    logger.debug("    Scanning in chunks for invalid values...", file=sys.stderr)

    flat = tensor.detach().view(-1)
    total_size = flat.numel()
    chunk_size = INT_MAX // 4  # ~0.5B elements per chunk to stay safe
    max_report = 10
    found = 0

    for chunk_start in range(0, total_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_size)
        chunk = flat[chunk_start:chunk_end]

        # Identify invalid entries in chunk
        invalid_mask = ~torch.isfinite(chunk)
        # out_of_range_mask = (~invalid_mask) & ((chunk < -1) | (chunk > 1))
        # num_out_of_range = out_of_range_mask.sum().item()
        # print(f"    Number of finite values not in [-1, 1]: {num_out_of_range}")
        # if num_out_of_range > 0:
        #     out_of_range_values = chunk[out_of_range_mask][:32]
        #     print("    First 32 finite values out of range:")
        #     print(out_of_range_values)
        if invalid_mask.any():
            bad_indices = torch.nonzero(invalid_mask, as_tuple=False).squeeze()
            logger.debug(f"  Chunk [{chunk_start}:{chunk_end}] has {bad_indices.numel()} invalid entries:", file=sys.stderr)
            for idx in bad_indices:
                global_idx = chunk_start + idx.item()
                val = chunk[idx].item()
                logger.debug(f"    [flat index {global_idx}] value: {val}", file=sys.stderr)
                found += 1
                if found >= max_report:
                    raise ValueError(f"NaN or Inf detected in tensor: {name}")
    raise ValueError(f"NaN or Inf detected in tensor: {name}")

class KDTrainer(Trainer):
    def __init__(self, teacher_model, loss_type, mean_prob=0, kd_loss_top_k=0, ranking_type="none", ranking_R=32, ranking_beta=10000, *args, **kwargs):
        """
        Args:
            loss_type: Type of loss to use. Choices: ["reverse", "forward", "tlsd", "cakld", "cakld_ranking", "jsd"].
            mean_prob: Mean probability for knowledge distillation.
            kd_loss_top_k: If > 0, use top-k logits for knowledge distillation. Else use all logits.
            ranking_R: Top-R for ranking loss.
        """
        if FSDP_DEBUG:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        super().__init__(*args, **kwargs)
        # self.tlsd = tsld_loss
        self.loss_fct_none = torch.nn.CrossEntropyLoss(reduction="none")
        self.tmp = 1
        self.teacher_model = teacher_model
        # self.reverse_loss = reverse_loss
        self.loss_type = loss_type
        self.mean_prob = mean_prob
        self.ce_loss_none = CrossEntropyLoss(reduction="none")
        self._mse_loss = MSELoss()

        self.kd_loss_partial = not (kd_loss_top_k == 0)
        self.topk = kd_loss_top_k
        self.weight_mode = "dcg"  # "dcg" or "power_law"

        self.ranking_type = ranking_type
        self.ranking_R = ranking_R  # Top-R for ranking loss

        self.ranking_loss_func = lambda **_: 0.0
        self.ranking_beta = ranking_beta
        if self.ranking_type == "dcg_pair_logistic":
            self.ranking_loss_func = self.ranking_dcg_pair_logistic


    @staticmethod
    def ranking_dcg_pair_logistic(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        # Get ranking of logits
        B, L, V = student_logits.shape
        R = self.ranking_R
        student_flat = student_logits.view(B*L, V)[:, :R]
        teacher_flat = teacher_logits.view(B*L, V)[:, :R]

        # Step 1: teacher ranks (1 = best)
        sorted_idx = torch.argsort(teacher_flat, dim=-1, descending=True)
        ranks = torch.empty_like(sorted_idx, dtype=torch.long)
        ranks.scatter_(1, sorted_idx, torch.arange(1, R + 1, device=teacher_flat.device).unsqueeze(0).expand_as(sorted_idx))

        # Step 2: pairwise comparisons per row
        s_i = student_flat.unsqueeze(2)  # [N, R, 1]
        s_j = student_flat.unsqueeze(1)  # [N, 1, R]
        r_i = ranks.unsqueeze(2)         # [N, R, 1]
        r_j = ranks.unsqueeze(1)         # [N, 1, R]

        better_mask = (r_i < r_j)        # [N, R, R]
        margin = s_i - s_j               # [N, R, R]

        # Step 3: top-heavy weights
        a = 1.0 / torch.log2(ranks.float() + 1.0)  # [N, R]
        w_ij = torch.where(better_mask, a.unsqueeze(2), a.unsqueeze(1))  # [N, R, R]

        if FSDP_DEBUG:
            torch.cuda.memory._dump_snapshot(f"/mnt/external/cuda_mem_snapshot/cakld_ranking_Rank{int(os.environ.get('LOCAL_RANK', 0))}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        # Step 4: pairwise logistic loss
        loss_mat = torch.nn.functional.softplus(-margin)  # [N, R, R]
        loss_per_row = (loss_mat * w_ij * better_mask).sum(dim=(1, 2)) / better_mask.sum(dim=(1, 2)).clamp_min(1)   # [N]
        loss_per_row = loss_per_row.mean()

        return loss_per_row

        
    def cakld_loss(self, labels, student_logits: torch.Tensor, teacher_logits: torch.Tensor, beta_prob):
        mask = (labels != -100)

        if self.kd_loss_partial:
            teacher_logits, top_k_indices = teacher_logits.topk(self.topk, dim=-1)
            student_logits = student_logits.gather(-1, top_k_indices)

        ranking_loss = self.ranking_loss_func(self, student_logits, teacher_logits)

        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
        student_output_log_prob = F.log_softmax(student_logits, dim=2)
        reverse_kl = F.kl_div(teacher_output_log_prob, student_output_log_prob, reduction="none", log_target=True).sum(-1)  # [batch_size, seq_len]
        forward_kl = F.kl_div(student_output_log_prob, teacher_output_log_prob, reduction="none", log_target=True).sum(-1)
        kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
        kl_loss *= mask
        average_kl_loss = kl_loss.sum(-1).mean()

        if BITDISTILLER_DEBUG:
            for name, param in self.model.named_parameters():
                if name == "lm_head.weight":
                    temp = param.flatten()
                    indicess = [252242294, 252242428, 252242438, 252242864, 252243310, 252243468, 252243794, 252244596, 252244804, 252245202]
                    for idx in indicess:
                        logger.debug(f"{name}: {idx} = {temp[idx]}")
                if param.requires_grad:
                    param.register_hook(lambda grad, name=name: check_for_nan_or_inf(grad, f"{name}.grad"))
                    check_for_nan_or_inf(param, name)
            check_for_nan_or_inf(student_logits, "student_logits")
            check_for_nan_or_inf(teacher_logits, "teacher_logits")

            teacher_output_log_prob.requires_grad_()
            teacher_output_log_prob.register_hook(lambda grad: check_for_nan_or_inf(grad, "teacher_output_log_prob.grad"))
            check_for_nan_or_inf(teacher_output_log_prob, "teacher_output_log_prob")

            student_output_log_prob.register_hook(lambda grad: check_for_nan_or_inf(grad, "student_output_log_prob.grad"))
            check_for_nan_or_inf(student_output_log_prob, "student_output_log_prob")

            reverse_kl.register_hook(lambda grad: check_for_nan_or_inf(grad, "reverse_kl.grad"))
            check_for_nan_or_inf(reverse_kl, "reverse_kl")

            forward_kl.register_hook(lambda grad: check_for_nan_or_inf(grad, "forward_kl.grad"))
            check_for_nan_or_inf(forward_kl, "forward_kl")

            kl_loss.register_hook(lambda grad: check_for_nan_or_inf(grad, "kl_loss.grad"))
            check_for_nan_or_inf(kl_loss, "kl_loss")

        torch.cuda.empty_cache()  # Clear cache to avoid memory issues
        return average_kl_loss + self.ranking_beta * ranking_loss

    def jsd_loss(self, labels, student_logits, teacher_logits, beta_prob):
        mask = (labels != -100)
        student_prob = F.softmax(student_logits, dim=2)
        teacher_prob = F.softmax(teacher_logits, dim=2)

        c_prob = beta_prob * teacher_prob + (1-beta_prob) * student_prob
        c_log_prob = c_prob.log()


        kl_loss_f = beta_prob * F.kl_div(c_log_prob, teacher_prob, reduction="none")
        kl_loss_r = (1 - beta_prob) * F.kl_div(c_log_prob, student_prob, reduction="none")
        kl_loss = kl_loss_f + kl_loss_r

        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()

        return kl_loss

    def ce_loss(self, labels, student_logits, teacher_logits):
        mask = (labels != -100)

        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_log_prob = F.log_softmax(teacher_logits / self.tmp, dim=2)

        # loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        kl_loss = F.kl_div(model_output_log_prob, real_output_log_prob, reduction="none", log_target=True)
        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss

    def re_loss(self, labels, student_logits, teacher_logits):
        mask = (labels != -100)

        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
        student_output_log_prob = F.log_softmax(student_logits, dim=2)

        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        kl_loss = F.kl_div(teacher_output_log_prob, student_output_log_prob, reduction="none", log_target=True)
        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss

    def TLSD_loss(self, labels, student_logits, teacher_logits):
        shift_logits = student_logits[..., :-1, :].contiguous() 
        tc_shift_logits = teacher_logits[..., :-1, :].contiguous() 

        # Step 1. get per-token ce loss with teacher logits
        tc_shift_labels = labels[..., 1:].contiguous().to(labels.device)
        tc_loss_all = self.ce_loss_none(tc_shift_logits.view(-1,tc_shift_logits.size(-1)), tc_shift_labels.view(-1))

        # Step 2. get token-scale with tc_loss_all and temperatured softmax function
        tc_all = tc_loss_all.reshape(tc_shift_logits.shape[0], -1)
        token_scale = torch.nn.functional.softmax(tc_all / 10, dim=-1).clone().detach()

        # Step 3. logit distillation with token-scale
        student_likelihood = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(tc_shift_logits, dim=-1)
        tsld_loss = (torch.sum((- targets_prob * student_likelihood), dim=-1) * token_scale).sum() # SUM

        return tsld_loss

    def mse_loss(self, student_logits, teacher_logits):
        return self._mse_loss(student_logits, teacher_logits)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        log_fsdp_debug(logger, f"Allocated memory before forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        labels = inputs.pop('labels', None)  # remove labels from inputs to avoid passing them to the model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        log_fsdp_debug(logger, f"Allocated memory after teacher forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        # forward pass
        log_fsdp_debug(logger, f"Allocated memory before student forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
        student_outputs = model(**inputs)
        log_fsdp_debug(logger, f"Allocated memory after student forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
        student_logits = student_outputs.get("logits")

        if not return_outputs:
            del student_outputs

        # torch.save(student_logits, "/root/model/acr_duda/code/rej_analysis/sft_student_logits.pt")
        # torch.save(teacher_logits, "/root/model/acr_duda/code/rej_analysis/sft_teacher_logits.pt")
        # torch.save(inputs, "/root/model/acr_duda/code/rej_analysis/sft_inputs.pt")
        # raise 1

        kd_loss = 0.0
        if model.kd_loss_scale > 0.0:
            if self.loss_type == "reverse":
                kd_loss = self.re_loss(labels, student_logits, teacher_logits)
            elif self.loss_type == "forward":
                kd_loss = self.ce_loss(labels, student_logits, teacher_logits)
            elif self.loss_type == "tlsd":
                kd_loss = self.TLSD_loss(labels, student_logits, teacher_logits)
            elif self.loss_type == "cakld":
                kd_loss = self.cakld_loss(labels, student_logits, teacher_logits, self.mean_prob)
            elif self.loss_type == "cakld_ranking":
                kd_loss = self.cakld_loss_with_ranking(labels, student_logits, teacher_logits, self.mean_prob)
            elif self.loss_type == "jsd":
                kd_loss = self.jsd_loss(labels, student_logits, teacher_logits, 0.5)
        log_fsdp_debug(logger, f"Allocated memory after loss computation: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
        del teacher_logits
        del student_logits

        tok_loss = model.kd_loss_scale * kd_loss
        return (tok_loss, student_outputs) if return_outputs else tok_loss
    
    