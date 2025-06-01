import os
import json
import csv

# Base path and checkpoints
base_path = "/mnt/external/checkpoints/Qwen/Qwen3-8B/1b-random-r0.5_cakld_ctx4096"
checkpoints = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 3809]
target_benchmarks = ['aime24', 'gpqa:diamond', 'math_500']

# Initialize results dict
benchmark_scores = {bench: [] for bench in target_benchmarks}
valid_checkpoints = []

# Process each checkpoint
for ckpt in checkpoints:
    ckpt_path = os.path.join(base_path, f"checkpoint-{ckpt}")
    evals_path = os.path.join(ckpt_path, "evals", "results")

    if not os.path.exists(evals_path):
        print(f"Warning: evals path not found for checkpoint {ckpt}")
        continue

    subfolders = [d for d in os.listdir(evals_path) if os.path.isdir(os.path.join(evals_path, d))]
    if len(subfolders) != 1:
        print(f"Warning: expected 1 folder in {evals_path}, found {len(subfolders)}.")
        continue

    results_dir = os.path.join(evals_path, subfolders[0])
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    scores = {bench: "NA" for bench in target_benchmarks}
    found_benchmarks = set()

    for json_file in json_files:
        file_path = os.path.join(results_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            results = data.get("results", {})
            for key, val in results.items():
                if "|" in key:
                    parts = key.split("|")
                    if len(parts) >= 2:
                        bench_name = parts[1]
                        if bench_name in target_benchmarks and bench_name not in found_benchmarks:
                            score = val.get("extractive_match")
                            if isinstance(score, float):
                                scores[bench_name] = f"{score:.6f}"
                                found_benchmarks.add(bench_name)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    for bench in target_benchmarks:
        benchmark_scores[bench].append(scores[bench])
    valid_checkpoints.append(ckpt)

# Write transposed CSV
output_csv = os.path.join(base_path, "scores_aime.csv")
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['benchmark'] + [str(ckpt) for ckpt in valid_checkpoints])
    for bench in target_benchmarks:
        writer.writerow([bench] + benchmark_scores[bench])

print(f"\nDone. Scores written to: {output_csv}")
