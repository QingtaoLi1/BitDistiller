from contextlib import contextmanager
from typing import Iterable
import deepspeed
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import torch


@contextmanager
def maybe_zero3():
    if is_deepspeed_zero3_enabled():
        with deepspeed.zero.Init():
            if deepspeed.comm.get_rank() == 0:
                yield
    else:
        yield

@contextmanager
def maybe_zero3_gather(params: torch.nn.Parameter | Iterable[torch.nn.Parameter]):
    if is_deepspeed_zero3_enabled():
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                yield
    else:
        yield