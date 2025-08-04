import json
from datasets import load_dataset
from tqdm import tqdm


### https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
# code: 10,108,883
# math: 22,066,397
# science: 708,920
# instruction following: 56,339
# chat: 39,792
# safety: 31,426

ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT")
code = ds["code"]
block_size = 500000
block_num = (len(code) + block_size - 1) // block_size
for block_id in range(block_num):
    start_idx = block_id * block_size
    end_idx = min(start_idx + block_size, len(code))
    print(f"Exporting block {block_id + 1}/{block_num} with {end_idx - start_idx} samples...")
    output_path = f"/mnt/external/data/nemotron/code/nemotron-sft-code_500K_block_{block_id}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(start_idx, end_idx), total=end_idx - start_idx, desc="Exporting samples"):
            rec = code[i]
            # if len(rec["output"]) > 64 * 1024:  # Skip outputs longer than 64K
            #     continue
            pair = [[rec["input"][0]['content'], rec["output"]]]
            _ = f.write(json.dumps(pair) + "\n")

math = ds["math"]
block_size = 500000
block_num = (len(math) + block_size - 1) // block_size
for block_id in range(block_num):
    start_idx = block_id * block_size
    end_idx = min(start_idx + block_size, len(math))
    print(f"Exporting block {block_id + 1}/{block_num} with {end_idx - start_idx} samples...")
    output_path = f"/mnt/external/data/nemotron/math/nemotron-sft-math_500K_block_{block_id}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(start_idx, end_idx), total=end_idx - start_idx, desc="Exporting samples"):
            rec = math[i]
            # if len(rec["output"]) > 64 * 1024:  # Skip outputs longer than 64K
            #     continue
            pair = [[rec["input"][0]['content'], rec["output"]]]
            _ = f.write(json.dumps(pair) + "\n")
