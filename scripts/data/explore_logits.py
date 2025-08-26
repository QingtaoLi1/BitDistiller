import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_logits(path: str):
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    elif path.endswith(".pt"):
        logits = torch.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    logger.info(f"Logits loaded from {path}.")
    return logits.to(torch.float32)


def analyze_logits(logits, lower_bound=-40.0, upper_bound=70.0, bucket_size=0.125, eps=1e-5):
    """
    Analyze the logit value distribution by counting the number of values within each bucket.
    Set a bucket every 0.125, e.g. [0, 0.125), [0.125, 0.25), ..., [99.875, 100).
    """
    num_buckets = round((upper_bound - lower_bound) / bucket_size)
    logits = logits.view(-1).cuda()
    min_val = logits.min().item()
    max_val = logits.max().item()
    logger.info(f"Logits min: {min_val}, max: {max_val}")

    # Create buckets
    buckets = torch.zeros(num_buckets, dtype=torch.int64)
    
    # buck_indices = torch.floor(logits / bucket_size - bucket_size / 2 + eps)
    # for i in tqdm(range(num_buckets)):
    #     buckets[i] = (buck_indices == i).sum()

    for i in tqdm(range(num_buckets)):
        bucket_lower_bound = lower_bound + i * bucket_size
        bucket_upper_bound = lower_bound + (i + 1) * bucket_size
        buckets[i] = ((logits >= bucket_lower_bound) & (logits < bucket_upper_bound)).sum()

    return buckets


def plot_buckets(x_range: list, buckets0: torch.Tensor, buckets1: torch.Tensor, buckets2: torch.Tensor):
    plt.figure(figsize=(12, 6))
    plt.bar(x_range, buckets0.numpy(), width=0.12, alpha=0.4, label='BF16')
    plt.bar(x_range, buckets1.numpy(), width=0.12, alpha=0.4, label='Init')
    plt.bar(x_range, buckets2.numpy(), width=0.12, alpha=0.4, label='Step_900')
    plt.xlabel('Logit Value Buckets')
    plt.ylabel('Count')
    plt.title('Logit Value Distribution')
    plt.legend()
    # plt.show()
    plt.savefig(f"buckets/logit_distribution_{i}.png")
    plt.close()

def plot_diff_buckets(x_range, buckets0: torch.Tensor, buckets1: torch.Tensor, buckets2: torch.Tensor):
    plt.figure(figsize=(12, 6))
    plt.bar(x_range, buckets0.numpy(), width=0.12, alpha=0.4, label='Diff = Step900 - BF16')
    plt.bar(x_range, buckets1.numpy(), width=0.12, alpha=0.4, label='Diff = Step900 - Init')
    plt.bar(x_range, buckets2.numpy(), width=0.12, alpha=0.4, label='Diff = Init - BF16')
    plt.xlabel('Logit Value Buckets')
    plt.ylabel('Count')
    plt.title('Logit Value Distribution (Difference)')
    plt.legend()
    # plt.show()
    plt.savefig(f"buckets/logit_distribution_diff_{i}.png")
    plt.close()

def plot_probs(probs0: torch.Tensor, probs1: torch.Tensor, probs2: torch.Tensor):
    # # Sort by descending probs1
    # probs1, sort_idx = torch.sort(probs1, descending=True)
    # probs2 = probs2[sort_idx]

    plt.figure(figsize=(12, 6))
    # plt.bar(range(len(probs1)), probs1.cpu().numpy(), width=0.2, alpha=0.8, label='Init', color="blue")
    # plt.bar(range(len(probs2)), probs2.cpu().numpy(), width=0.2, alpha=0.8, label='Step_900', color="orange")
    plt.plot(range(len(probs0)), probs0.cpu().numpy(), marker='s', label='BF16', color="green")
    plt.plot(range(len(probs1)), probs1.cpu().numpy(), marker='o', label='Init', color="blue")
    plt.plot(range(len(probs2)), probs2.cpu().numpy(), marker='^', label='Step_900', color="red")
    plt.xlabel('Tokens')
    plt.ylabel('Probability')
    plt.title('Ranking-1 Probability Distribution')
    plt.legend()
    # plt.show()
    plt.savefig(f"probs/prob_distribution_rank1_token128_{i}.png")
    plt.close()

def plot_probs_top2(probs01: torch.Tensor, probs02: torch.Tensor, probs21: torch.Tensor, probs22: torch.Tensor):
    # # Sort by descending probs1
    # probs1, sort_idx = torch.sort(probs1, descending=True)
    # probs2 = probs2[sort_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(probs01)), probs01.cpu().numpy(), marker='s', label='BF16', color="blue")
    plt.plot(range(len(probs02)), probs02.cpu().numpy(), marker='s', label='Init', color="blue", linestyle='dashed')
    plt.plot(range(len(probs21)), probs21.cpu().numpy(), marker='^', label='Step_900', color="red")
    plt.plot(range(len(probs22)), probs22.cpu().numpy(), marker='^', label='Step_900', color="red", linestyle='dashed')
    plt.xlabel('Tokens')
    plt.ylabel('Probability')
    plt.title('Ranking-1/2 Probability Distribution')
    plt.legend()
    # plt.show()
    plt.savefig(f"probs/prob_distribution_rank12_token128_{i}.png")
    plt.close()


def draw_buckets(i: int, path0: str, path1: str, path2: str):
    logger.info(f"Starting logits {i} loading...")
    logits0 = load_logits(path0)
    logits1 = load_logits(path1)
    logits2 = load_logits(path2)

    buckets0 = analyze_logits(logits0)
    buckets1 = analyze_logits(logits1)
    buckets2 = analyze_logits(logits2)
    os.makedirs("buckets", exist_ok=True)
    torch.save(buckets0, f"buckets/buckets_BF16_{i}.pt")
    torch.save(buckets1, f"buckets/buckets_Init_{i}.pt")
    torch.save(buckets2, f"buckets/buckets_Step900_{i}.pt")

    x_range = [round(-40.0 + i * 0.125, 3) for i in range(880)]
    plot_buckets(x_range[240:400], buckets0[240:400], buckets1[240:400], buckets2[240:400])


def draw_logits_diff(i: int, path0: str, path1: str, path2: str):
    logger.info(f"Starting logits {i} loading...")
    logits0 = load_logits(path0)
    logits1 = load_logits(path1)
    logits2 = load_logits(path2)

    diff_step_bf16 = logits2 - logits0
    diff_step_init = logits2 - logits1
    diff_init_bf16 = logits1 - logits0
    buckets_step_bf16 = analyze_logits(diff_step_bf16)
    buckets_step_init = analyze_logits(diff_step_init)
    buckets_init_bf16 = analyze_logits(diff_init_bf16)

    os.makedirs("buckets", exist_ok=True)
    torch.save(buckets_step_bf16, f"buckets/diff_buckets_step_bf16_{i}.pt")
    torch.save(buckets_step_init, f"buckets/diff_buckets_step_init_{i}.pt")
    torch.save(buckets_init_bf16, f"buckets/diff_buckets_init_bf16_{i}.pt")

    x_range = [round(-40.0 + i * 0.125, 3) for i in range(880)]
    plot_diff_buckets(x_range[240:400], buckets_step_bf16[240:400], buckets_step_init[240:400], buckets_init_bf16[240:400])


def draw_probs(i: int, path0: str, path1: str, path2: str):
    logger.info(f"Starting probs {i} loading...")
    logits0 = load_logits(path0)
    logits1 = load_logits(path1)
    logits2 = load_logits(path2)

    prob0 = torch.softmax(logits0, dim=-1, dtype=torch.float32)[0]
    prob1 = torch.softmax(logits1, dim=-1, dtype=torch.float32)[0]
    prob2 = torch.softmax(logits2, dim=-1, dtype=torch.float32)[0]

    prob0_topk, indices = prob0.topk(128)
    prob1_topk = torch.gather(prob1, 1, indices)
    prob2_topk = torch.gather(prob2, 1, indices)

    os.makedirs("probs", exist_ok=True)
    torch.save(prob0_topk, f"probs/probs_top128_bf16_{i}.pt")
    torch.save(prob1_topk, f"probs/probs_top128_init_{i}.pt")
    torch.save(prob2_topk, f"probs/probs_top128_step900_{i}.pt")

    plot_probs(prob0_topk[:32, 0], prob1_topk[:32, 0], prob2_topk[:32, 0])
    plot_probs_top2(prob0_topk[:32, 0], prob0_topk[:32, 1], prob2_topk[:32, 0], prob2_topk[:32, 1])

    # torch.set_printoptions(linewidth=180, sci_mode=False, edgeitems=0)
    #
    # def pretty_tensor(t: torch.Tensor, width=6, precision=4):
    #     return "[" + ", ".join(f"{x:{width}.{precision}f}" for x in t.tolist()) + "]"
    #
    # NUM_TOKENS = 64
    # for i in range(NUM_TOKENS):
    #     print(f"BF logits: {pretty_tensor(logits1[0][i].topk(16).values)}")
    #     print(f"V1 logits: {pretty_tensor(logits2[0][i].topk(16).values)}")
    #     print(f"BF prob: {pretty_tensor(prob1[0][i].topk(16).values)}")
    #     print(f"V1 prob: {pretty_tensor(prob2[0][i].topk(16).values)}")



for i in range(10):
    path0 = f"/home/qingtaoli/mnt/logits/Qwen/Qwen3-14B/BF16/sample_{i}_logits.pt"
    path1 = f"/home/qingtaoli/mnt/logits/Qwen/Qwen3-14B/Init/sample_{i}_logits.pt"
    path2 = f"/home/qingtaoli/mnt/logits/Qwen/Qwen3-14B/V1_16k/checkpoint-900/sample_{i}_logits.pt"
    
    draw_buckets(i, path0, path1, path2)
    draw_logits_diff(i, path0, path1, path2)
    draw_probs(i, path0, path1, path2)

## Next:
# 1. See logits subtraction (+ or - ?)
# 2. See prob distribution
# 3. See top-k logits distribution
