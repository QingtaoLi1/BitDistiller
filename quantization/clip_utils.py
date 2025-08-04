#TODO Dataset Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import __version__ as transformers_version
from packaging import version
if version.parse(transformers_version) >= version.parse("4.41.0"):
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
else:
    # a random but not model type
    Phi3ForCausalLM = int
if version.parse(transformers_version) >= version.parse("4.37.0"):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
else:
    Qwen2ForCausalLM = int
if version.parse(transformers_version) >= version.parse("4.50.0"):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
else:
    Gemma3ForConditionalGeneration = int
if version.parse(transformers_version) >= version.parse("4.51.0"):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
else:
    Qwen3ForCausalLM = int
import torch
from datasets import load_dataset
import random
import json
import os
import torch.nn as nn

def get_calib_dataset(datasets="pileval", tokenizer=None, n_samples=128, block_size=1024):
    if datasets == "pile":
        return get_pile_dataset(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size)
    elif datasets == "gsm8k":
        return get_calib_dataset_gsm8k(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size)
    elif datasets == "code":
        return get_calib_dataset_code(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size)
    elif datasets == "merged_gmc":
        return get_calib_dataset_json_dataset(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size, data_path="/mnt/external/data/nemotron/nemotron-sft-math-char64k_500k.jsonl")
    elif datasets == "openr1_math":
        return get_calib_dataset_json_dataset(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size, data_path="/mnt/external/data/OpenR1-Math-220k/default.json")
    elif datasets == "nemotron_code":
        return get_calib_dataset_nemotron(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size, split="code")
    elif datasets == "nemotron_math":
        return get_calib_dataset_nemotron(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size, split="math")
    elif datasets == "epicoder":
        return get_calib_dataset_epicoder(tokenizer=tokenizer, n_samples=n_samples, block_size=block_size)

def get_pile_dataset(tokenizer=None, n_samples=512, block_size=512):
    # dataset = load_dataset("json", data_files="/root/model/llm-awq/val.jsonl.zst", split="train")
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.map(lambda x: {
        'text': x['text']
    })
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0

    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

# TODO: Don't do spliting when code and gsm8k
def get_calib_dataset_code(tokenizer=None, n_samples=512, block_size=512):
    # dataset = load_dataset("json", data_files="/root/model/datasets/code/EvolInstruct-Code-80k.json", split="train")
    dataset = load_dataset("json", data_files="nickrosh/Evol-Instruct-Code-80k-v1", split="train")

    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0

    for data in dataset:
        istr = data["instruction"]
        opt = data["output"]
        line = f"Instruction:\n{istr}\nOutput:\n"
        line += opt
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_calib_dataset_gsm8k(tokenizer=None, n_samples=512, block_size=512):
    # download from here: https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl
    data_path = "/root/model/gsm8k-ScRel/data/train_use.jsonl"

    with open(data_path, 'r') as f:
        dataset_for_eval = f.readlines()

    dataset = [json.loads(item.strip()) for item in dataset_for_eval]
    random.seed(42)
    dataset = random.sample(dataset, k=min(n_samples * 10, len(dataset)))
    samples = []
    n_run = 0

    for data in dataset:
        istr = data["query"]
        opt = data["response"]
        line = f"Instruction:\n{istr}\nOutput:\n"
        line += opt
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_calib_dataset_json_dataset(tokenizer=None, n_samples=512, block_size=512, data_path=None):
    if data_path is None:
        raise ValueError("data_path must be provided for clip calibration dataset!")

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = f.readlines()

    samples = []
    n_run = 0

    for data in dataset:
        item = json.loads(data.strip())
        istr = item[0][0]
        opt = item[0][1]
        line = f"{istr}\n\n{opt}"
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_calib_dataset_nemotron(tokenizer=None, n_samples=512, block_size=512, split=None):
    if split is None:
        raise ValueError("split must be provided for nemotron clip calibration dataset!")

    dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT", split=split)
    samples = []
    n_run = 0

    for data in dataset:
        istr = data["input"][0]['content']
        opt = data["output"]
        line = f"{istr}\n\n{opt}"
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

def get_calib_dataset_epicoder(tokenizer=None, n_samples=512, block_size=512):
    dataset = load_dataset("microsoft/EpiCoder-func-380k", split="train")
    samples = []
    n_run = 0

    for data in dataset:
        istr = data["instruction"]
        opt = data["output"]
        line = f"{istr}\n\n<think></think>\n{opt}"
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, Phi3ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, Qwen2ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, Qwen3ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, Gemma3ForConditionalGeneration):
        layers = model.language_model.model.layers
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif isinstance(model, Phi3ForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, Qwen2ForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, Qwen3ForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, Gemma3ForConditionalGeneration):
        model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def build_model_and_enc(model_path):
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    enc = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs)

    model.eval()

    return model, enc

@torch.no_grad()
def apply_clip(module, clip_results):
    if not isinstance(clip_results, list):
        clip_list = clip_results["clip"]
    else:
        clip_list = clip_results
    if len(clip_list) == 0:
        return
    if len(clip_list[0]) == 3:
        for name, max_val, min_val in clip_list:
            layer = get_op_by_name(module, name)
            layer.cuda()
            max_val = max_val.to(layer.weight.device)
            min_val = min_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
    else:
        raise 1