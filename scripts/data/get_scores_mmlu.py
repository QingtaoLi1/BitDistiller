import os
import re
import csv


benchmark_list = {
    "/mnt/external/checkpoints/Qwen/Qwen3-14B/merged_gmc_90k_90k_90k_cakld_ctx4096_cosine":             list(range(100,2100,100)) + [2109],
    "/mnt/external/checkpoints/Qwen/Qwen3-14B/merged_gmc_90k_90k_90k_cakld_ctx4096_cosine_cycle300":    list(range(100,2100,100)) + [2109],
    "/mnt/external/checkpoints/Qwen/Qwen3-14B/nemotron_code_char16k_cakld_ctx4096_epoch4":              list(range(100,3000,100)),
}


def extract_scores(base_path, checkpoints):
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

    print(f"Done. MMLU scores saved to '{output_csv}'. You can open it directly in Excel.\n")



if __name__ == "__main__":
    for base_path, checkpoints in benchmark_list.items():
        print(f"Processing {base_path} with checkpoints {checkpoints}")
        extract_scores(base_path, checkpoints)
