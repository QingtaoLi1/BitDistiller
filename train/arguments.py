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
    kd_loss_top_k: int = field(
        default=0,
        metadata={"help": "Top K logits to calculate loss. 0 means all logits."}
    )
    cakld_steps: int = field(
        default=10,
        metadata={"help": "How many step to caculate the coefficient of CAKLD."}
    )

    mkld_warmup_steps: int = field(
        default=50,
        metadata={"help": "Number of warmup steps (warmup means using cakld loss) for MKLD loss."}
    )
    mkld_reverse: bool = field(
        default=False,
        metadata={"help": "Whether to reverse the MKLD loss masks."}
    )

    ranking_type: str = field(
        default="none",
        metadata={"help": "Type of ranking loss to use. Should be one of [`none`, `dcg_pair_logistic`]."}
    )
    ranking_R: int = field(
        default=32,
        metadata={"help": "Top-R for ranking loss."}
    )
    ranking_beta: float = field(
        default=10000,
        metadata={"help": "Beta parameter for ranking loss."}
    )

    use_teacher_entropy_coeff: bool = field(
        default=False,
        metadata={"help": "Whether to use teacher-entropy as token-level KD coefficient."}
    )

    token_curriculum: Optional[str] = field(
        default=None,
        metadata={"help": "The token-level teacher-entropy curriculum learning setting. "
                  "If not set, no curriculum learning is used. Choices: [None, 'const', 'linear', 'exponential']. Default: None."}
    )
    token_curriculum_min: Optional[float] = field(
        default=None,
        metadata={"help": "The minimum value of the token-level curriculum threshold. "
                  "When exponential is used, this value will be used as the starting point. Default: None."}
    )
    token_curriculum_max: Optional[float] = field(
        default=None,
        metadata={"help": "The maximum value of the token-level curriculum threshold. Default: None."}
    )
    token_curriculum_exp_base: Optional[float] = field(
        default=None,
        metadata={"help": "The exponential base for exponential curriculum learning. Default: None."}
    )
    token_curriculum_end_step: Optional[int] = field(
        default=None,
        metadata={"help": "The step at which the token-level curriculum reaches its maximum threshold. Default: None."}
    )

    wasserstein_sinkhorn_reg: float = field(
        default=0.1,
        metadata={"help": "The regularization coefficient for Sinkhorn algorithm in Wasserstein loss."}
    )
    wasserstein_num_iters: int = field(
        default=10,
        metadata={"help": "The number of iterations for Sinkhorn algorithm in Wasserstein loss."}
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
    may_resume: bool = field(
        default=False,
        metadata={"help": "Whether to resume training from the last checkpoint."}
    )
