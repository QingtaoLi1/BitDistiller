import sys
sys.path.append("../quantization")
from qlinear import convertModelToQuant
from clip_utils import apply_clip

import io
import json
import logging
import os
import random
from tqdm import tqdm
from typing import Dict

import torch
from torch.distributed.fsdp import fully_shard, FSDPModule, ShardingStrategy, CPUOffload, MixedPrecisionPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data import DataLoader

import transformers
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from mytrainer import KDTrainer
from data_loading import make_supervised_data_module
from debug import add_nan_inf_hooks
from arguments import ModelArguments, DataArguments, TrainingArguments


os.environ['NCCL_DEBUG'] = 'ERROR'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
from logger import BITDISTILLER_DEBUG, FSDP_DEBUG, log_info, log_bitdistiller_debug, log_fsdp_debug
logger = logging.getLogger(__name__)


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        logger.info("Tokenizer has not eos token")
    return tokenizer

def get_model_kwargs(model_args: ModelArguments, training_args: TrainingArguments):
    n_gpus = torch.cuda.device_count()
    max_memory = f'80000MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if "34B" in model_args.model_name_or_path:
        device_map = None

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    if training_args.use_flash_attn:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    model_kwargs = {
        "max_memory": max_memory,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
    }
    return model_kwargs


def train():
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    random.seed(TrainingArguments.seed)
    model_kwargs = get_model_kwargs(model_args, training_args)

    tokenizer = get_tokenizer(model_args, training_args)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.train_kd:
        logger.info("loading Teacher Model...")
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=False,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
        fsdp_kwargs = {
            "reshard_after_forward": True,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        }
        for layer in teacher_model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(teacher_model.model, **fsdp_kwargs)
        fully_shard(teacher_model, **fsdp_kwargs)
        # teacher_model.cuda()
        # teacher_model = FSDP(teacher_model,
        #                      sharding_strategy=ShardingStrategy.FULL_SHARD,
        #                      cpu_offload=CPUOffload(offload_params=True))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=teacher_model,
            )
        logger.info("Teacher Model loaded")

        mean_prob=0
        if training_args.kd_loss_type == "cakld":
            logger.info("Get the main Prob!")
            probDataloader = DataLoader(
                data_module['train_dataset'],
                shuffle=False,
                collate_fn=data_module['data_collator'],
                batch_size=training_args.per_device_train_batch_size,
                drop_last=False,
            )

            prob = 0
            for step, batch in tqdm(enumerate(probDataloader)):
                if step > training_args.cakld_steps:
                    break
                batch = {k: v.to(teacher_model.device) for k, v in batch.items()}
                batch.pop("labels", None)
                device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
                log_fsdp_debug(logger, f"Coeff step {step}: before allocated memory = {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
                with torch.no_grad():
                    outputs = teacher_model(**batch)
                log_fsdp_debug(logger, f"Coeff step {step}: after allocated memory = {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
                logits = outputs.get("logits").contiguous()
                prob1 = torch.nn.functional.softmax(logits, dim=-1)
                prob1 = torch.max(prob1, dim=-1).values
                prob += prob1.mean()
                del logits, outputs
            mean_prob = prob / training_args.cakld_steps
            mean_prob = torch.Tensor(mean_prob.to(teacher_model.device))
            dist.all_reduce(mean_prob, op=dist.ReduceOp.SUM)
            mean_prob = mean_prob / dist.get_world_size()
            logger.info(f"Get the coefficient: {mean_prob}")

    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        **model_kwargs
    )

    if training_args.quant_type is not None:
        logger.info("Converting the model to qat, this may take a while...")
        model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type=training_args.quant_type, q_group_size=training_args.q_group_size)

    if training_args.clip is not None:
        logger.info(f"Loading pre-computed Clipping results from {training_args.clip}")
        clip_results = torch.load(training_args.clip)
        apply_clip(model, clip_results)
        logger.info("Clipping init successfully!")
    model.config.use_cache = False
    model.kd_loss_scale = 1.0

    if tokenizer.pad_token is None:
        logger.info("Tokenizer has not padding token")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if BITDISTILLER_DEBUG:    
        add_nan_inf_hooks(model)
        # hook_last_hidden(model)

    logger.info("Waiting for other processes to synchronize...")
    torch.distributed.barrier()
    if int(os.environ.get('LOCAL_RANK', '0')) == 0:
        logger.info(f"Training_args: {training_args}")

    logger.info("Starting trainer...")
    if training_args.train_kd:
        trainer = KDTrainer(model=model, tokenizer=tokenizer, teacher_model=teacher_model,
                            loss_type=training_args.kd_loss_type, kd_loss_top_k=training_args.kd_loss_top_k,
                            ranking_type=training_args.ranking_type, ranking_R=training_args.ranking_R,
                            ranking_beta=training_args.ranking_beta,
                            mean_prob=mean_prob, args=training_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if training_args.may_resume and get_last_checkpoint(training_args.output_dir) is not None:
        logger.info(f"Resuming training from checkpoint: {get_last_checkpoint(training_args.output_dir)}")
        trainer.train(resume_from_checkpoint=get_last_checkpoint(training_args.output_dir))
    else:
        logger.info(f"Starting training from scratch.")
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()