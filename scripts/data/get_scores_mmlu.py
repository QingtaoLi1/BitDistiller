import os
import re
import ast
import csv

# Base path (adjust if different from previous)
base_path = "/mnt/external/checkpoints/Qwen/Qwen3-8B/1b-grad-l3-r0.5_nokd_ctx4096"
checkpoints = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 3809]

# Output CSV
output_csv = os.path.join(base_path, "scores_mmlu.csv")

# Store scores
mmlu_scores = []
valid_checkpoints = []

for ckpt in checkpoints:
    log_path = os.path.join(base_path, f"checkpoint-{ckpt}", "MMLU.log")
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found.")
        continue

    with open(log_path, 'r') as f:
        content = f.read()

    match = re.search(r"\{'mmlu-acc':\s*([0-9.]+)\}", content)
    if match:
        score = float(match.group(1))
        mmlu_scores.append(f"{score:.6f}")
        valid_checkpoints.append(ckpt)
    else:
        print(f"'mmlu-acc' not found in {log_path}")

# Write CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric'] + [str(ckpt) for ckpt in valid_checkpoints])
    writer.writerow(['mmlu-acc'] + mmlu_scores)

print(f"\nDone. MMLU scores saved to '{output_csv}'. You can open it directly in Excel.")
