from datasets import load_dataset
from itertools import zip_longest
import json
from tqdm import tqdm


# for split in ["default", "extended", "all"]:
for name in ["default"]:
    ds = load_dataset("open-r1/OpenR1-Math-220k", name)["train"]

    samples = []
    dsl = ds.to_list()
    print(len(dsl))

    for index, s in tqdm(enumerate(dsl)):
        # Suggested by https://huggingface.co/datasets/open-r1/OpenR1-Math-220k#dataset-curation.
        problem = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + s['problem']
        correctness_math_verify = s['correctness_math_verify']
        correctness_llama = [] if s['correctness_llama'] is None else s['correctness_llama']
        generations = s['generations']
        for verified_math, verified_llama, generation in zip_longest(correctness_math_verify, correctness_llama, generations, fillvalue=None):
            if verified_math or verified_llama:
                samples.append([problem, generation])
                break
    print(len(samples))

    with open(f"/home/qingtaoli/OpenR1-Math-220k/{name}.json", 'w', encoding="utf-8") as f:
        for s in tqdm(samples):
            f.write(json.dumps(s) + "\n")
