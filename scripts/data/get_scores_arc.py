import os
import re
import ast
import csv

# Define base path and checkpoints
base_path = "/mnt/external/checkpoints/Qwen/Qwen3-8B/1b-grad-l3-r0.5_cakld_ctx4096"
checkpoints = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 3809]
benchmarks = ['arc_challenge', 'hellaswag', 'piqa', 'winogrande']

# Output CSV file
output_csv = os.path.join(base_path, "scores_arc.csv")

# Initialize a dictionary for benchmark → list of accs
benchmark_accs = {bench: [] for bench in benchmarks}

# Gather acc values
valid_checkpoints = []
for ckpt in checkpoints:
    log_path = os.path.join(base_path, f"checkpoint-{ckpt}", "arc.log")
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found.")
        continue

    with open(log_path, 'r') as f:
        content = f.read()

    match = re.search(r"\{'results':\s*\{.*?\}\s*,\s*'versions':\s*\{.*?\}\s*,\s*'config':\s*\{.*?\}\}", content, re.DOTALL)

    if match:
        try:
            data = ast.literal_eval(match.group(0))
            valid_checkpoints.append(ckpt)
            for b in benchmarks:
                acc = data['results'].get(b, {}).get('acc', 'NA')
                if isinstance(acc, float):
                    benchmark_accs[b].append(f"{acc:.6f}")
                else:
                    benchmark_accs[b].append("NA")
        except Exception as e:
            print(f"Failed to parse checkpoint {ckpt}: {e}")
    else:
        print(f"No results found in checkpoint {ckpt}")

# Write transposed CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['benchmark'] + [str(ckpt) for ckpt in valid_checkpoints])
    for bench in benchmarks:
        writer.writerow([bench] + benchmark_accs[bench])

print(f"\nDone. Transposed results written to '{output_csv}'. Open it in Excel as a standard CSV.")
