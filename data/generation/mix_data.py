import json
import random

all_outputs = []

DATA_ROOT = "/mnt/sdb1/qingtaoli/data-phi-3-wiki300kalpaca5k/"
json_path1 = DATA_ROOT + "/wikitext-2-generated/alpaca_T0.7_N1024_S42_5000.json"
json_path2 = DATA_ROOT + "/wikitext-2-generated/wikitext_T0.7_N1024_S42_300000.json"
# DATA_ROOT = "/home/superbench/qingtaoli/data-llama-2-7b"
# json_path1 = DATA_ROOT + "/wikitext-2-generated/wikitext_T0.7_N1024_S42_3000.json"
# json_path2 = DATA_ROOT + "/wikitext-2-generated/alpaca_T0.7_N1024_S42_5000.json"

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
with open(DATA_ROOT + '/wikitext-2-generated/mix_wiki_alpaca_8000.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')
