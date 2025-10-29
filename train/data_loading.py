import copy
from dataclasses import dataclass, field
import json
import logging
import random
import os
from tqdm import tqdm
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import transformers


IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


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
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, sources: Sequence[str], targets: Sequence[str], split: str):
        super().__init__()

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
        sources = [instance['input_ids'] for instance in instances]
        targets = [instance['labels'] for instance in instances]

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
    with open(data_args.data_path, 'r') as f:
        lines = f.readlines()
    all_dataset = [json.loads(line.strip()) for line in tqdm(lines, desc=f"Loading dataset: {data_args.data_path}")]

    sources, targets = zip(*[(s[0][0], f"{s[0][1]}{tokenizer.eos_token}") for s in all_dataset])

    dataset_size = len(sources)
    if data_args.max_train_samples < dataset_size:
        indices = random.sample(range(dataset_size), data_args.max_train_samples)
        sources, targets = [sources[i] for i in indices], [targets[i] for i in indices]

    train_dataset = SupervisedDataset(sources=sources, targets=targets, split="train")
    eval_dataset = SupervisedDataset(sources=sources, targets=targets, split="eval")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

