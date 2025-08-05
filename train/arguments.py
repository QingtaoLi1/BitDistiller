import json
from dataclasses import dataclass, field
import transformers
from typing import Optional, Dict


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
