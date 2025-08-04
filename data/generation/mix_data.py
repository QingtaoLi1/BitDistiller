import json
import random
import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("--data_path", default="", type=str, help="model path")
parser.add_argument("--dataset_name1", default="", type=str, help="name of the datasets #1")
parser.add_argument("--dataset_name2", default="", type=str, help="name of the datasets #2")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--max_sample1", type=int, default=None, help="max_sample #1")
parser.add_argument("--max_sample2", type=int, default=None, help="max_sample #2")
parser.add_argument("--temperature", type=int, default=0.7, help="generation temperature")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="max new tokens")
args = parser.parse_args()


all_outputs = []

# DATA_ROOT = "/mnt/sdb1/qingtaoli/data-phi-3-wiki300kalpaca5k/"
# json_path1 = DATA_ROOT + "/wikitext-2-generated/alpaca_T0.7_N1024_S42_5000.json"
# json_path2 = DATA_ROOT + "/wikitext-2-generated/wikitext_T0.7_N1024_S42_300000.json"
DATA_ROOT = args.data_path
json_path1 = DATA_ROOT + f"/{args.dataset_name1}_T{args.temperature}_N{args.max_new_tokens}_S{args.seed}_{args.max_sample1}.json"
json_path2 = DATA_ROOT + f"/{args.dataset_name2}_T{args.temperature}_N{args.max_new_tokens}_S{args.seed}_{args.max_sample2}.json"

with open(json_path1, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

with open(json_path2, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

random.shuffle(all_outputs)

# with open('/home/superbench/qingtaoli/data/wikitext-2-generated/mix_wiki_alpaca_8000.json', 'w') as f:
with open(DATA_ROOT + f'/mix_{args.dataset_name1}_{args.dataset_name2}_{args.max_sample1 + args.max_sample2}.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')
