from accelerate.utils import merge_fsdp_weights
import logging
import os
import shutil
import time
import torch
import transformers


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


folder_sizes = {
    # "Qwen3-14B": 59079507460,
    "Qwen3-14B": 59078800000,
}


def merge_fsdp_shards_on_gpu(checkpoint_dir: str, output_file: str):
    import glob
    import torch
    from safetensors.torch import save_file
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU.")
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    device = torch.device("cuda", 0)

    shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "__*_*")))
    if not shard_files:
        logger.info(f"No FSDP shard files found in '{checkpoint_dir}'.")
        logger.info("Please check the path and the file naming pattern.")
        return
    logger.info(f"Found {len(shard_files)} shard files in '{checkpoint_dir}'.")

    consolidated_state_dict = {}
    logger.info("Merging shards on GPU...")
    for shard_file in tqdm(shard_files, desc="Loading shards"):
        # The magic happens here: map_location=device loads the shard
        # directly into GPU memory, avoiding the CPU bottleneck.
        shard_state_dict = torch.load(shard_file, map_location=device)
        
        # The shard contains a 'state_dict' key with the actual model weights
        if 'state_dict' in shard_state_dict:
            consolidated_state_dict.update(shard_state_dict['state_dict'])
        else:
            # Handle cases where the shard file is the state_dict itself
            consolidated_state_dict.update(shard_state_dict)
    logger.info("Saving the merged model...")
    save_file(consolidated_state_dict, output_file)
    
    logger.info(f"✅ Successfully merged model and saved to '{output_file}'.")


def check_az_storage_files_integrity(file_path: str, expected_size: int) -> bool:
    """Check if a folder exists and matches the expected size."""
    if not os.path.exists(file_path):
        return False
    actual_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(file_path) for filename in filenames)
    return actual_size >= expected_size

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="A faster, GPU-based script to merge FSDP sharded checkpoints."
    # )
    # parser.add_argument(
    #     "--checkpoint_dir",
    #     type=str,
    #     required=True,
    #     help="Path to the directory containing FSDP model shards.",
    # )
    # parser.add_argument(
    #     "--output_file",
    #     type=str,
    #     required=True,
    #     help="Path to save the output .safetensors file.",
    # )
    # args = parser.parse_args()

    mount_root = ""
    az_container_url = ""
    az_container_sas = ""
    
    origin_path = f"{mount_root}/models/Qwen/Qwen3-14B/"
    origin_files = [f for f in os.listdir(origin_path) if f.endswith(".json") or f.endswith(".txt")]
    az_blob_path = "checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_mkld_F_50"
    az_blob_url = f"{az_container_url}/{az_blob_path}"
    mount_path = f"{mount_root}/{az_blob_path}/"

    for ckpt in range(50, 300, 50):
        in_url = f"{az_blob_url}/checkpoint-{ckpt}/pytorch_model_fsdp_0/"
        in_mount_path = os.path.join(mount_path, f"checkpoint-{ckpt}", "pytorch_model_fsdp_0")
        in_temp_path = f"./"
        out_temp_path = f"./hf/"
        out_url = f"{az_blob_url}/checkpoint-{ckpt}/"

        logger.info(f"Downloading checkpoint-{ckpt} from {in_url} to {in_temp_path}...")
        while not check_az_storage_files_integrity(in_mount_path, folder_sizes["Qwen3-14B"]):
            logger.warning(f"Checkpoint-{ckpt} not ready at {in_mount_path}, will check again after 20min...")
            time.sleep(1200)
        ret = os.system(f'azcopy copy "{in_url}{az_container_sas}" "{in_temp_path}" --recursive')

        in_temp_path = os.path.join(in_temp_path, "pytorch_model_fsdp_0")
        logger.info(f"Converting FSDP model from {in_temp_path} to HF format at {out_temp_path}...")
        merge_fsdp_weights(in_temp_path, out_temp_path, safe_serialization=True)
        # merge_fsdp_shards_on_gpu(in_temp_path, out_temp_path)
        logger.info(f"Loading FP32 model...")
        for f in origin_files:
            shutil.copy(os.path.join(origin_path, f), os.path.join(out_temp_path, f))
        model = transformers.AutoModelForCausalLM.from_pretrained(out_temp_path, torch_dtype=torch.bfloat16, device_map="auto")
        logger.info(f"Saving BF16 model...")
        model.save_pretrained(out_temp_path)
        del model
        logger.info(f"Removing FP32 model file...")
        os.remove(os.path.join(out_temp_path, "model.safetensors"))
        logger.info(f"Uploading HF model from {out_temp_path} to {out_url}...")
        ret = os.system(f'azcopy copy "{out_temp_path}" "{out_url}{az_container_sas}" --recursive')
        if ret != 0:
            raise RuntimeError(f"Failed to upload HF model from {out_temp_path} to {out_url}")
        
        logger.info(f"Removing temporary files...")
        shutil.rmtree(in_temp_path)
        shutil.rmtree(out_temp_path)
        logger.info(f"All done!")
