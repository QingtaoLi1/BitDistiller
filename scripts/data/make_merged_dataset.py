import random
import os


def interleaved_merge_sample(lists, counts, seed=None):
    if seed is not None:
        random.seed(seed)

    assert len(lists) == len(counts)
    total = sum(counts)
    
    # Sample from each list, preserving internal order
    sampled = []
    for lst, count in zip(lists, counts):
        assert count <= len(lst), "Sample count exceeds list length"
        indices = sorted(random.sample(range(len(lst)), count))
        sampled.append([lst[i] for i in indices])

    # Create iterators
    iterators = [iter(s) for s in sampled]
    current_counts = [0] * len(lists)
    max_counts = counts.copy()

    # Compute cycle pattern
    base = min(counts)
    gcd = lambda a, b: a if b == 0 else gcd(b, a % b)
    from math import gcd as math_gcd
    ratio_gcd = math_gcd(math_gcd(counts[0], counts[1]), counts[2])
    unit_pattern = [c // ratio_gcd for c in counts]  # e.g., [2, 3, 1] for 400/600/200

    # Build interleaved pattern
    merged = []
    while any(current_counts[i] < max_counts[i] for i in range(len(lists))):
        for i, num in enumerate(unit_pattern):
            for _ in range(num):
                if current_counts[i] < max_counts[i]:
                    merged.append(next(iterators[i]))
                    current_counts[i] += 1
    return merged


with open("D:\\data\\data_efficacy\\processed_1b-grad-l3-r0.5.jsonl", 'r', encoding="utf-8") as f:
    general = f.readlines()
    print(f"General dataset size: {len(general)}")
with open("D:\\data\\OpenR1-Math-220k\\default.json", 'r', encoding="utf-8") as f:
    math = f.readlines()
    print(f"Math dataset size: {len(math)}")
with open("D:\\data\\EpiCoder-func-380k\\epicoder-func-380k.jsonl", 'r', encoding="utf-8") as f:
    code = f.readlines()
    print(f"Code dataset size: {len(code)}")

merged = interleaved_merge_sample(
    lists=[general, math, code],
    counts=[90000, 90000, 90000],
    seed=42
)
print(f"Total merged items: {len(merged)}")

save_path = "D:\\data\\merged_dataset\\merged_dataset.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding="utf-8") as f:
    for item in merged:
        f.write(item.strip() + "\n")
print("Merged dataset saved to D:\\data\\merged_dataset\\merged_dataset.jsonl")

# list1 = list(range(100, 600))  # 500 items
# list2 = list(range(2000, 3000))  # 1000 items
# list3 = list(range(5000, 5500))  # 500 items
# merged = interleaved_merge_sample(
#     lists=[list1, list2, list3],
#     counts=[400, 600, 200],
#     seed=42
# )
# for i in range(0, 60, 6):
#     print([merged[i + j] for j in range(6)])
