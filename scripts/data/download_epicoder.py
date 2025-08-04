import json
from datasets import load_dataset
from tqdm import tqdm

def export_to_jsonl(output_path: str = "epicoder-func-380k.jsonl"):
    ds = load_dataset("microsoft/EpiCoder-func-380k", split="train")
    total = len(ds)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in tqdm(ds, total=total, desc="Exporting samples"):
            pair = [[rec["instruction"], rec["output"]]]
            f.write(json.dumps(pair) + "\n")

    print("✅ Export complete.")

if __name__ == "__main__":
    export_to_jsonl("epicoder-func-380k.jsonl")
