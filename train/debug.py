import logging
import os
import torch


logger = logging.getLogger(__name__)

INT_MAX = 2_147_483_647


def check_for_nan_or_inf(tensor, name=""):
    if int(os.environ.get('LOCAL_RANK', '0')) != 0:
        return
    
    if not torch.is_floating_point(tensor):
        return  # skip int/bool tensors

    if torch.isfinite(tensor).all():
        return

    logger.debug(f"\n[!] NaN or Inf detected in: {name}")
    logger.debug(f"    Shape: {tensor.shape}, Device: {tensor.device}")
    logger.debug("    Scanning in chunks for invalid values...")

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
        if invalid_mask.any():
            bad_indices = torch.nonzero(invalid_mask, as_tuple=False).squeeze()
            logger.debug(f"  Chunk [{chunk_start}:{chunk_end}] has {bad_indices.numel()} invalid entries:")
            for idx in bad_indices:
                global_idx = chunk_start + idx.item()
                val = chunk[idx].item()
                logger.debug(f"    [flat index {global_idx}] value: {val}")
                found += 1
                if found >= max_report:
                    raise ValueError(f"NaN or Inf detected in tensor: {name}")
    raise ValueError(f"NaN or Inf detected in tensor: {name}")

def add_nan_inf_hooks(model : torch.nn.Module):
    for name, module in model.named_modules():
        def forward_hook(mod, input, output, name=name):
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            if isinstance(output, torch.Tensor):
                check_for_nan_or_inf(output, name=f"{name} (forward output)")
            elif isinstance(output, (tuple, list)):
                for idx, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        check_for_nan_or_inf(out, name=f"{name} (forward output {idx})")

        def backward_hook(mod, grad_input, grad_output, name=name):
            if isinstance(grad_output, torch.nn.Module):
                for name, tensor in model.state_dict().items():
                    if "weight" in name:
                        check_for_nan_or_inf(tensor, name=f"{name} (backward weight)")
            for idx, grad in enumerate(grad_input):
                if isinstance(grad, torch.Tensor):
                    check_for_nan_or_inf(grad, name=f"{name} (backward grad_input {idx})")
            for idx, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    check_for_nan_or_inf(grad, name=f"{name} (backward grad_output {idx})")

        # nn.modules.module.register_module_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

def hook_last_hidden(model : torch.nn.Module):
    for name, module in model.named_modules():
        if name.endswith("transformer.h.35") or name.endswith("model.layers.35"):
            logger.debug(f"Registering backward hook on: {name}")
            def bwd_hook(mod, ginp, goutp):
                for i, g in enumerate(ginp):
                    if isinstance(g, torch.Tensor):
                        check_for_nan_or_inf(g, name=f"{name} (backward grad_input {i})")
                for i, g in enumerate(goutp):
                    if isinstance(g, torch.Tensor):
                        check_for_nan_or_inf(g, name=f"{name} (backward grad_output {i})")
            module.register_full_backward_hook(bwd_hook)

