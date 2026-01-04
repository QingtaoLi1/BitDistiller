import ast
import csv
import json
import os
import re


# Define base path and checkpoints
benchmark_list = {
    "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_token_curriculum_te_batch8":    list(range(100,9700,100)),
}
benchmarks_arc = ['arc_challenge', 'hellaswag', 'piqa', 'winogrande']
target_benchmarks_aime = ['aime24', 'gpqa:diamond', 'math_500', 'lcb:codegeneration', 'ifeval']
benchmark_metric_key = {
    'aime24': 'extractive_match',
    'gpqa:diamond': 'extractive_match',
    'math_500': 'extractive_match',
    'lcb:codegeneration': 'codegen_pass@1',
    'ifeval': 'prompt_level_strict_acc',      # Usually use this
}


def extract_scores_arc(base_path, checkpoints):
    # Output CSV file
    output_csv = os.path.join(base_path, "scores_arc.csv")

    # Initialize a dictionary for benchmark → list of accs
    benchmark_accs = {bench: [] for bench in benchmarks_arc}

    # Gather acc values
    for ckpt in checkpoints:
        log_path = os.path.join(base_path, f"checkpoint-{ckpt}", "hf", "arc.log")
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found.")
            for b in benchmarks_arc:
                benchmark_accs[b].append("")
            continue

        with open(log_path, 'r') as f:
            content = f.read()

        match = re.search(r"\{'results':\s*\{.*?\}\s*,\s*'versions':\s*\{.*?\}\s*,\s*'config':\s*\{.*?\}\}", content, re.DOTALL)

        if match:
            try:
                data = ast.literal_eval(match.group(0))
                for b in benchmarks_arc:
                    acc = data['results'].get(b, {}).get('acc', '')
                    if isinstance(acc, float):
                        benchmark_accs[b].append(f"{acc:.6f}")
                    else:
                        benchmark_accs[b].append("")
            except Exception as e:
                print(f"Failed to parse checkpoint {ckpt}: {e}")
                for b in benchmarks_arc:
                    benchmark_accs[b].append("")
        else:
            print(f"No results found in checkpoint {ckpt}")
            for b in benchmarks_arc:
                benchmark_accs[b].append("")

    # Write transposed CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['benchmark'] + [str(ckpt) for ckpt in checkpoints])
        for bench in benchmarks_arc:
            writer.writerow([bench] + benchmark_accs[bench])

    print(f"Done. Transposed results written to '{output_csv}'. Open it in Excel as a standard CSV.\n")

def extract_scores_mmlu(base_path, checkpoints):
    # Output CSV
    output_csv = os.path.join(base_path, "scores_mmlu.csv")

    # Store scores
    mmlu_scores = []

    for ckpt in checkpoints:
        log_path = os.path.join(base_path, f"checkpoint-{ckpt}", "hf", "MMLU.log")
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found.")
            mmlu_scores.append("")
            continue

        with open(log_path, 'r') as f:
            content = f.read()

        match = re.search(r"\{'mmlu-acc':\s*([0-9.]+)\}", content)
        if match:
            score = float(match.group(1))
            mmlu_scores.append(f"{score:.6f}")
        else:
            print(f"'mmlu-acc' not found in {log_path}")
            mmlu_scores.append("")

    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric'] + [str(ckpt) for ckpt in checkpoints])
        writer.writerow(['mmlu-acc'] + mmlu_scores)

    print(f"Done. MMLU scores saved to '{output_csv}'. You can open it directly in Excel.\n")

def extract_scores_aime(base_path, checkpoints):
    # Initialize results dict
    benchmark_scores = {bench: [] for bench in target_benchmarks_aime}

    # Process each checkpoint
    for index, ckpt in enumerate(checkpoints):
        ckpt_path = os.path.join(base_path, f"checkpoint-{ckpt}", "hf")
        evals_path = os.path.join(ckpt_path, "evals", "results")

        scores = {bench: "NA" for bench in target_benchmarks_aime}

        if not os.path.exists(evals_path):
            print(f"Warning: evals path not found for checkpoint {ckpt}")
        else:
            subfolders = [d for d in os.listdir(evals_path) if os.path.isdir(os.path.join(evals_path, d))]

            # Check all subfolders
            for subfolder in subfolders:
                results_dir = os.path.join(evals_path, subfolder)
                json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
                # Reorder json_files from old to new according to the timestamp in its name.
                # The file name is like "results_2025-10-24T20-39-28.670870.json"
                json_files.sort(key=lambda x: x)
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
                                    if bench_name in target_benchmarks_aime:
                                        metric_key = benchmark_metric_key[bench_name]
                                        score = val.get(metric_key)
                                        if isinstance(score, float):
                                            scores[bench_name] = f"{score:.6f}"
                                            # found_benchmarks.add(bench_name)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")

        for bench in target_benchmarks_aime:
            benchmark_scores[bench].append(scores[bench])

    # Write transposed CSV
    output_csv = os.path.join(base_path, "scores_aime.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['benchmark'] + [str(ckpt) for ckpt in checkpoints])
        for bench in target_benchmarks_aime:
            writer.writerow([bench] + benchmark_scores[bench])

    print(f"Done. Scores written to: {output_csv}\n")



if __name__ == "__main__":
    for base_path, checkpoints in benchmark_list.items():
        print(f"Processing {base_path} with checkpoints {checkpoints}")
        extract_scores_arc(base_path, checkpoints)
        extract_scores_mmlu(base_path, checkpoints)
        extract_scores_aime(base_path, checkpoints)
