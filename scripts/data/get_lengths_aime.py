import os
import pandas as pd
import csv
import glob
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Base path and checkpoints
base_path = "/mnt/external/checkpoints/Qwen/Qwen3-8B/1b-grad-l3-r0.5_cakld_ctx4096"
checkpoints = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 3809]

# Sample size to benchmark name mapping
sample_size_to_benchmark = {
    480: "aime24",
    792: "gpqa:diamond",
    2000: "math_500"
}

# Collect results: benchmark -> ckpt -> list of lengths
results = defaultdict(lambda: defaultdict(list))

# Also store averages: benchmark -> ckpt -> avg
averages = defaultdict(dict)

for ckpt in tqdm(checkpoints):
    print(f"Processing checkpoint {ckpt}...")
    ckpt_path = os.path.join(base_path, f"checkpoint-{ckpt}")
    details_path = os.path.join(ckpt_path, "evals", "details")

    if not os.path.exists(details_path):
        print(f"Warning: details path not found for checkpoint {ckpt}")
        continue

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    except Exception as e:
        print(f"Error loading tokenizer from {ckpt_path}: {e}")
        continue

    # Locate the one-folder in details/
    detail_subdirs = [d for d in os.listdir(details_path) if os.path.isdir(os.path.join(details_path, d))]
    if len(detail_subdirs) != 1:
        print(f"Warning: expected 1 folder in {details_path}, found {len(detail_subdirs)}.")
        continue

    one_detail_path = os.path.join(details_path, detail_subdirs[0])
    benchmark_dirs = [d for d in os.listdir(one_detail_path) if os.path.isdir(os.path.join(one_detail_path, d))]

    for bench_dir in benchmark_dirs:
        parquet_files = glob.glob(os.path.join(one_detail_path, bench_dir, "*.parquet"))
        if not parquet_files:
            continue

        parquet_file = parquet_files[0]
        try:
            df = pd.read_parquet(parquet_file)
            lengths = []
            for i in range(len(df)):
                pred = df.at[i, "predictions"]
                try:
                    text = pred[0]
                    if isinstance(text, str):
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        lengths.append(len(tokens))
                except Exception:
                    continue
            num_samples = len(lengths)
            benchmark_name = sample_size_to_benchmark.get(num_samples)
            if benchmark_name:
                results[benchmark_name][ckpt] = lengths
                averages[benchmark_name][ckpt] = np.mean(lengths)
            else:
                print(f"Warning: Unknown benchmark for sample count {num_samples}")
        except Exception as e:
            print(f"Failed processing {parquet_file}: {e}")

# Output CSV with multi-level headers
output_csv = os.path.join(base_path, "scores_length_aime_all.csv")
benchmarks = list(sample_size_to_benchmark.values())

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header rows
    header_row_1 = ["sample_index"]
    header_row_2 = [""]
    for bench in benchmarks:
        for ckpt in checkpoints:
            header_row_1.append(bench)
            header_row_2.append(f"ckpt-{ckpt}")
    writer.writerow(header_row_1)
    writer.writerow(header_row_2)

    # Average row
    avg_row = ["average"]
    for bench in benchmarks:
        for ckpt in checkpoints:
            avg = averages[bench].get(ckpt)
            avg_row.append(f"{avg:.2f}" if avg is not None else "")
    writer.writerow(avg_row)

    # Determine max sample count
    max_len = 0
    for bench in benchmarks:
        for ckpt in checkpoints:
            max_len = max(max_len, len(results[bench].get(ckpt, [])))

    # Sample rows
    for i in range(max_len):
        row = [i]
        for bench in benchmarks:
            for ckpt in checkpoints:
                val_list = results[bench].get(ckpt, [])
                row.append(val_list[i] if i < len(val_list) else "")
        writer.writerow(row)

output_csv

