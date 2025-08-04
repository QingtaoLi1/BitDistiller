import pandas as pd
import csv
import os
import glob
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

# Target path (single model, special case)
model_base_path = "/mnt/external/models/Qwen/Qwen3-8B/evals/details/Qwen/Qwen3-8B"

# Sample size to benchmark name mapping
sample_size_to_benchmark = {
    480: "aime24",
    792: "gpqa:diamond",
    2000: "math_500"
}

# Load tokenizer from base model path
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer: {e}")

# Search all .parquet files under three subfolders
all_parquet_files = glob.glob(os.path.join(model_base_path, "*", "*.parquet"))

results = {}
averages = {}

for parquet_file in all_parquet_files:
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
            results[benchmark_name] = lengths
            averages[benchmark_name] = np.mean(lengths)
        else:
            print(f"Warning: Unknown benchmark for sample count {num_samples} in {parquet_file}")
    except Exception as e:
        print(f"Failed processing {parquet_file}: {e}")

# Output CSV
output_csv = "/mnt/external/models/Qwen/Qwen3-8B/scores_length_aime.csv"
benchmarks = list(sample_size_to_benchmark.values())

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["sample_index"] + benchmarks
    writer.writerow(header)

    # Write average
    avg_row = ["average"]
    for bench in benchmarks:
        avg = averages.get(bench)
        avg_row.append(f"{avg:.2f}" if avg is not None else "")
    writer.writerow(avg_row)

    # Write per-sample lengths
    max_len = max((len(v) for v in results.values()), default=0)
    for i in range(max_len):
        row = [i]
        for bench in benchmarks:
            val_list = results.get(bench, [])
            row.append(val_list[i] if i < len(val_list) else "")
        writer.writerow(row)


