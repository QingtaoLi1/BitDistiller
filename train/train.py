import sys
sys.path.append("../quantization")
from qlinear import QLinear, convertModelToQuant
from clip_utils import apply_clip
# from utils import maybe_zero3
# from ..quantization.qlinear import QLinear, convertModelToQuant
# from ..quantization.clip_utils import apply_clip
# from ..quantization.utils import maybe_zero3

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, BitsAndBytesConfig, default_data_collator
from datasets import load_dataset
import json
import logging
import glob
import torch.distributed as dist

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import deepspeed
from mytrainer import KDTrainer
import random
from tqdm import tqdm
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


os.environ['NCCL_DEBUG'] = 'ERROR'
os.environ['DEEPSPEED_LOG_LEVEL'] = 'WARNING'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
BITDISTILLER_DEBUG = os.environ.get('BITDISTILLER_DEBUG', '0') == '1'

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



def arg_dict(arg: str) -> Dict[str, float]:
    """Convert a JSON-like string of key-value pairs into a dictionary."""
    if not arg:
        return {}
    if arg.startswith("{") and arg.endswith("}"):
        return json.loads(arg)
    else:
        return dict(item.split("=") for item in arg.split(","))

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)
    bits: int = field(
        default=2,
        metadata={"help": "How many bits to use."}
    )
    q_group_size: int = field(
        default=128,
        metadata={"help": "Quantization Group Size."}
    )
    quant_type: str = field(
        default="int2-asym",
        metadata={"help": "Quantization data type to use. Should be one of `int2-asym` or `ste-n2f3`."} # see quantization/qlinear.py
    )
    clip: str = field(
        default=None,
        metadata={"help": "The path of clip cache"}
    )
    train_kd: bool = field(default=False, metadata={"help": 'Whether to use KD to QAT'})
    kd_tmp: int = field(
        default=1,
        metadata={"help": "Temperature of KD"}
    )
    kd_loss_type: str = field(
        default=None,
        metadata={"help": "Type of loss function when KD-QAT"}
    )
    cakld_steps: int = field(
        default=10,
        metadata={"help": "How many step to caculate the coefficient of CAKLD."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention"}
    )
    lr_scheduler_kwargs: Optional[arg_dict] = field(
        default=None,
        metadata={
            "help": "Additional kwargs for the learning rate scheduler. "
            "This is a JSON string that will be parsed into a dictionary."
        },
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_sample: int, split: str):
        super().__init__()

        with open(data_path, 'r') as f:
            lines = f.readlines()
        all_dataset = [json.loads(line.strip()) for line in lines]

        sources, targets = zip(*[(s[0][0], f"{s[0][1]}{tokenizer.eos_token}") for s in all_dataset])

        dataset_size = len(sources)
        max_sample = min(max_sample or dataset_size, dataset_size)
        if max_sample < dataset_size:
            indices = random.sample(range(dataset_size), max_sample)
            self.sources, self.targets = [sources[i] for i in indices], [targets[i] for i in indices]
        else:
            self.sources, self.targets = sources, targets

        split_num = min(len(self.sources) // 10, 10)
        if split == "train":
            self.sources, self.targets = self.sources[split_num:], self.targets[split_num:]
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                logger.info(f"Using {len(self.sources)} samples to train")

                logger.debug("Example Data")
                logger.debug("sources: \n", self.sources[0])
                logger.debug("targets: \n", self.targets[0])

        elif split == "eval":
            self.sources, self.targets = self.sources[:split_num], self.targets[:split_num]
            logger.info(f"Using {len(self.sources)} samples to evaluation")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="train")
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="eval")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


INT_MAX = 2_147_483_647

def check_for_nan_or_inf(tensor, name=""):
    if int(os.environ.get('LOCAL_RANK', '0')) != 0:
        return
    
    if not torch.is_floating_point(tensor):
        return  # skip int/bool tensors

    if torch.isfinite(tensor).all():
        return

    logger.debug(f"\n[!] NaN or Inf detected in: {name}")
    logger.debug(f"    Shape: {tensor.shape}, Device: {tensor.device}")
    logger.debug("    Scanning in chunks for invalid values...")

    flat = tensor.detach().view(-1)
    total_size = flat.numel()
    chunk_size = INT_MAX // 4  # ~0.5B elements per chunk to stay safe
    max_report = 10
    found = 0

    for chunk_start in range(0, total_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_size)
        chunk = flat[chunk_start:chunk_end]

        # Identify invalid entries in chunk
        invalid_mask = ~torch.isfinite(chunk)
        if invalid_mask.any():
            bad_indices = torch.nonzero(invalid_mask, as_tuple=False).squeeze()
            logger.debug(f"  Chunk [{chunk_start}:{chunk_end}] has {bad_indices.numel()} invalid entries:")
            for idx in bad_indices:
                global_idx = chunk_start + idx.item()
                val = chunk[idx].item()
                logger.debug(f"    [flat index {global_idx}] value: {val}")
                found += 1
                if found >= max_report:
                    raise ValueError(f"NaN or Inf detected in tensor: {name}")
    raise ValueError(f"NaN or Inf detected in tensor: {name}")

def add_nan_inf_hooks(model : torch.nn.Module):
    for name, module in model.named_modules():
        def forward_hook(mod, input, output, name=name):
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            if isinstance(output, torch.Tensor):
                check_for_nan_or_inf(output, name=f"{name} (forward output)")
            elif isinstance(output, (tuple, list)):
                for idx, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        check_for_nan_or_inf(out, name=f"{name} (forward output {idx})")

        def backward_hook(mod, grad_input, grad_output, name=name):
            if isinstance(grad_output, torch.nn.Module):
                for name, tensor in model.state_dict().items():
                    if "weight" in name:
                        check_for_nan_or_inf(tensor, name=f"{name} (backward weight)")
            for idx, grad in enumerate(grad_input):
                if isinstance(grad, torch.Tensor):
                    check_for_nan_or_inf(grad, name=f"{name} (backward grad_input {idx})")
            for idx, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    check_for_nan_or_inf(grad, name=f"{name} (backward grad_output {idx})")

        # nn.modules.module.register_module_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

def hook_last_hidden(model : torch.nn.Module):
    for name, module in model.named_modules():
        if name.endswith("transformer.h.35") or name.endswith("model.layers.35"):
            logger.debug(f"Registering backward hook on: {name}")
            def bwd_hook(mod, ginp, goutp):
                for i, g in enumerate(ginp):
                    if isinstance(g, torch.Tensor):
                        check_for_nan_or_inf(g, name=f"{name} (backward grad_input {i})")
                for i, g in enumerate(goutp):
                    if isinstance(g, torch.Tensor):
                        check_for_nan_or_inf(g, name=f"{name} (backward grad_output {i})")
            module.register_full_backward_hook(bwd_hook)

def train():
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    random.seed(TrainingArguments.seed)
    n_gpus = torch.cuda.device_count()
    max_memory = f'80000MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if "34B" in model_args.model_name_or_path:
        device_map = None

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    logger.info("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    pad_status = True
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has not padding token")
        pad_status = False
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    logger.info(f"Loading {model_args.model_name_or_path} model")
    model_loading_kwargs = dict()
    if not is_deepspeed_zero3_enabled():
        model_loading_kwargs["device_map"] = device_map
    if training_args.use_flash_attn:
        model_loading_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_loading_kwargs["attn_implementation"] = 'eager'


    if training_args.train_kd:
        logger.info("Loading Teacher Model...")
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=False,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            **model_loading_kwargs
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        if pad_status is False:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=teacher_model,
            )
        logger.info("Teacher Model loaded")

        mean_prob=0
        if training_args.kd_loss_type == "cakld":
            logger.info("Get the main Prob!")
            probDataloader = DataLoader(
                data_module['train_dataset'],
                shuffle=False,
                collate_fn=data_module['data_collator'],
                batch_size=training_args.per_device_train_batch_size,
                drop_last=False,
            )

            prob = 0
            for step, batch in tqdm(enumerate(probDataloader)):
                if step > training_args.cakld_steps:
                    break
                batch = {k: v.to(teacher_model.device) for k, v in batch.items()}
                batch.pop("labels", None)
                with deepspeed.zero.GatheredParameters(list(teacher_model.parameters()), modifier_rank=None):
                    with torch.no_grad():
                        outputs = teacher_model(**batch)
                logits = outputs.get("logits").contiguous()
                prob1 = torch.nn.functional.softmax(logits, dim=-1)
                prob1 = torch.max(prob1, dim=-1).values
                prob += prob1.mean()
            mean_prob = prob / training_args.cakld_steps
            mean_prob = torch.Tensor(mean_prob.to(teacher_model.device))
            dist.all_reduce(mean_prob, op=dist.ReduceOp.SUM)
            mean_prob = mean_prob / dist.get_world_size()
            logger.info(f"Get the coefficient: {mean_prob}")

    logger.info("Loading Student Model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        **model_loading_kwargs
    )
    logger.info("Student Model loaded")

    if training_args.quant_type is not None:
        logger.info("Converting the model to qat, this may take a while...")
        model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type=training_args.quant_type, q_group_size=training_args.q_group_size)

    if is_deepspeed_zero3_enabled():
        with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                if training_args.clip is not None:
                    # q_config = {
                    #     "zero_point": True,  # by default True
                    #     "q_group_size": training_args.q_group_size,  # whether to use group quantization
                    # }
                    logger.info(f"Loading pre-computed Clipping results from {training_args.clip}")
                    clip_results = torch.load(training_args.clip)
                    apply_clip(model, clip_results)
    logger.info("Clipping init successfully!")
    model.config.use_cache = False
    model.kd_loss_scale = 1.0

    if BITDISTILLER_DEBUG:    
        add_nan_inf_hooks(model)
        # hook_last_hidden(model)


    logger.info("Starting trainer...")
    if training_args.train_kd:
        trainer = KDTrainer(model=model, tokenizer=tokenizer, teacher_model=teacher_model, loss_type=training_args.kd_loss_type, mean_prob=mean_prob, args=training_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()