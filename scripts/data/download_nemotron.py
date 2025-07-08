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

ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT", split="code")
output_path = "nemotron-sft-code-char64k.jsonl"
max_samples = 500000
with open(output_path, "w", encoding="utf-8") as f:
    for i in tqdm(range(max_samples), total=max_samples, desc="Exporting samples"):
        rec = ds[i]
        if len(rec["output"]) > 64 * 1024:  # Skip outputs longer than 64K
            continue
        pair = [[rec["input"][0]['content'], rec["output"]]]
        _ = f.write(json.dumps(pair) + "\n")
