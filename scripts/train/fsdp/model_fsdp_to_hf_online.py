from accelerate.utils import merge_fsdp_weights
import argparse
import logging
import os
import shutil
import torch
import transformers


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_ckpt_list(ckpt_str: str) -> list[int]:
    """Parse a checkpoint list string which can be a Python-style list or range."""
    ckpt_str = ckpt_str.strip()
    if ckpt_str.startswith("range(") and ckpt_str.endswith(")"):
        # Avoid using eval directly on untrusted input
        return list(range(*map(int, ckpt_str[6:-1].split(","))))
    elif ckpt_str.startswith("[") and ckpt_str.endswith("]"):
        # Avoid using eval
        return list(map(int, ckpt_str[1:-1].split(",")))
    else:
        raise ValueError("Invalid checkpoint list format. Use a Python-style list or range().")

def parse_ckpt_list(ckpt_str: str) -> list[int]:
    """
    Parse a checkpoint list string and return a list of strings.
    
    Supports:
    - List:         "[1, 10, 20]"
    - Single range: "range(0, 10, 2)"
    - Multi-range:  "range(0, 5) + range(10, 15)"
    """
    
    def _parse_single_range(range_str: str) -> list[int]:
        """Helper function to parse a single 'range(...)' string."""
        range_str = range_str.strip()
        if not (range_str.startswith("range(") and range_str.endswith(")")):
            raise ValueError(f"Invalid range format. Expected 'range(...)', got: {range_str}")
        
        # Get content inside range(...)
        content = range_str[6:-1].strip()
        if not content:
             raise ValueError("range() cannot be empty")

        # Split, strip whitespace from each arg, and convert to int
        try:
            args = [int(arg.strip()) for arg in content.split(",")]
            return list(range(*args))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid arguments in range '{range_str}': {e}")

    ckpt_str = ckpt_str.strip()
    int_result: list[int] = []
    
    if ckpt_str.startswith("[") and ckpt_str.endswith("]"):
        # Handle list: "[1, 2, 3]"
        content = ckpt_str[1:-1].strip()
        if not content:
            int_result = []  # Handle empty list "[]"
        else:
            try:
                # Split by comma, strip whitespace, and convert to int
                int_result = [int(x.strip()) for x in content.split(",")]
            except ValueError as e:
                raise ValueError(f"Invalid list format: {ckpt_str}") from e
            
    elif "+" in ckpt_str:
        # Handle multiple ranges: "range(...)+range(...)"
        parts = ckpt_str.split('+')
        final_list = []
        for part in parts:
            final_list.extend(_parse_single_range(part))
        int_result = final_list

    elif ckpt_str.startswith("range(") and ckpt_str.endswith(")"):
        # Handle single range: "range(...)"
        int_result = _parse_single_range(ckpt_str)

    else:
        raise ValueError("Invalid checkpoint list format. Use '[...]', 'range(...)', or 'range(...)+range(...)'.")
    
    return int_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A faster, GPU-based script to merge FSDP sharded checkpoints."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the directory containing FSDP model shards.",
    )
    parser.add_argument(
        "--ckpts",
        type=parse_ckpt_list,
        required=True,
        help="The checkpoint numbers to process. Can be a python-style list in \"[]\", or \"range(start,end,step)\".",
    )
    args = parser.parse_args()

    origin_path = "/home/qingtaoli/mnt/models/Qwen/Qwen3-14B/"
    origin_files = [f for f in os.listdir(origin_path) if f.endswith(".json") or f.endswith(".txt")]
    ckpt_path = args.ckpt_dir
    ckpt_list = args.ckpts

    for ckpt in ckpt_list:
        in_path = f"{ckpt_path}/checkpoint-{ckpt}/pytorch_model_fsdp_0/"
        out_path = f"{ckpt_path}/checkpoint-{ckpt}/hf/"

        logger.info(f"Converting FSDP model from {in_path} to HF format at {out_path}...")
        merge_fsdp_weights(in_path, out_path, safe_serialization=True)
        # merge_fsdp_shards_on_gpu(in_temp_path, out_temp_path)
        logger.info(f"Loading FP32 model...")
        for f in origin_files:
            # ret = os.system(f'azcopy copy "{origin_path}{f}{az_blob_sas}" "{out_temp_path}{f}"')
            shutil.copy(os.path.join(origin_path, f), os.path.join(out_path, f))
        
        # set device_map to cpu to avoid OOM
        model = transformers.AutoModelForCausalLM.from_pretrained(out_path, torch_dtype=torch.bfloat16)
        logger.info(f"Saving BF16 model...")
        model.save_pretrained(out_path)
        del model
        logger.info(f"Removing FP32 model file...")
        os.remove(os.path.join(out_path, "model.safetensors"))
        logger.info(f"Finished checkpoint-{ckpt} conversion.")
    logger.info(f"All done!")



