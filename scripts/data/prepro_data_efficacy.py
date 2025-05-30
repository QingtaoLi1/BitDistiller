import json
import os
from tqdm import tqdm

PATH = "/mnt/external/data/data_efficacy/"
FILE_NAME = "1b-random-r0.5.jsonl"
def read_jsonl(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)

def main():
    file_path = os.path.join(PATH, FILE_NAME)
    data = read_jsonl(file_path)

    samples = []
    for item in tqdm(data):
        if 'input' in item and 'text' in item:
            source = item['input']
            target = item['text']
            samples.append([[source, target]])
        else:
            print("Missing 'input' or 'text' in the item.")
            exit()

    # Repeat the first 10 samples as BitDistiller eval samples
    samples = samples[:10] + samples

    with open(os.path.join(PATH, f"processed_{FILE_NAME}"), 'w', encoding="utf-8") as f:
        for s in tqdm(samples):
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
