from accelerate.utils import merge_fsdp_weights
import logging
import os
import shutil
import torch
import transformers


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


origin_path = "/home/qingtaoli/mnt/models/Qwen/Qwen3-14B/"
origin_files = [f for f in os.listdir(origin_path) if f.endswith(".json") or f.endswith(".txt")]

for ckpt in [1300]:
    fsdp_path = f"/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_ranking32/checkpoint-{ckpt}/pytorch_model_fsdp_0/"
    hf_path = f"/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_ranking32/checkpoint-{ckpt}/hf/"
    logger.info(f"Converting FSDP model from {fsdp_path} to HF format at {hf_path}...")
    merge_fsdp_weights(fsdp_path, hf_path, safe_serialization=True)
    logger.info(f"Done.")
    logger.info(f"Loading FP32 model...")
    for f in origin_files:
        shutil.copy(os.path.join(origin_path, f), os.path.join(hf_path, f))
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16, device_map="auto")
    logger.info(f"Saving BF16 model...")
    model.save_pretrained(hf_path)
    logger.info(f"Removing FP32 model file...")
    os.remove(os.path.join(hf_path, "model.safetensors"))
    logger.info(f"All done!")
