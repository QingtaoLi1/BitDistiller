from datasets import load_dataset
from itertools import zip_longest
from tqdm import tqdm


for split in ["default", "extended", "all"]:
    ds = load_dataset("open-r1/OpenR1-Math-220k", split)["train"]

    samples = []
    dsl = ds.to_list()
    print(len(dsl))

    for index, s in tqdm(enumerate(dsl)):
        problem = s['problem']
        correctness_math_verify = s['correctness_math_verify']
        correctness_llama = [] if s['correctness_llama'] is None else s['correctness_llama']
        generations = s['generations']
        for verified_math, verified_llama, generation in zip_longest(correctness_math_verify, correctness_llama, generations, fillvalue=None):
            if verified_math or verified_llama:
                samples.append([problem, generation])
                break
    print(len(samples))

    with open(f"~/OpenR1-Math-220k/{split}.json", 'w', encoding="utf-8") as f:
        for s in tqdm(samples):
            b = '[[' + ', '.join(f'"{repr(item)}"' for item in s) + ']]'
            f.write(f'{b}\n')
