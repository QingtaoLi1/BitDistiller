import argparse
import torch
import math
from datasets import load_dataset
# CHANGED: Import standard HF classes instead of GPTQModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Evaluate Standard HF Model Perplexity on WikiText-2")
parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="/home/qingtaoli/models/tman/Qwen-8B", # Example default
    help="Path to the Standard HF model directory",
)
parser.add_argument(
    "--seq_len",
    type=int,
    required=False,
    default=4096,
    help="Sequence length for evaluation",
)
parser.add_argument(
    "--stride",
    type=int,
    required=False,
    default=64,
    help="Stride size for sliding window",
)
args = parser.parse_args()

# 1. Load Model (CHANGED)
logger.info(f"Loading model from: {args.model_path}")
model_path = args.model_path

# Load Tokenizer separately
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load Standard Model
# device_map="auto" will automatically use GPU if available and handles OOM better
# torch_dtype="auto" ensures we load in fp16/bf16 if supported, rather than fp32
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    dtype="auto",
    trust_remote_code=True
)

# Ensure the model is in eval mode
model.eval()

# 2. Load and Preprocess Data
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Merge all text into one long stream. 
logger.info("Tokenizing entire dataset...")
encodings = tokenizer("".join(ds["text"]), return_tensors="pt")

# Define window size. 
seq_len = args.seq_len  
stride = args.stride 

input_ids = encodings.input_ids
max_length = model.config.max_position_embeddings
seq_len = min(seq_len, max_length)

nlls = []
prev_end_loc = 0

logger.info(f"Starting evaluation with Sequence Length: {seq_len} and Stride: {stride}\n")

# 3. Evaluation Loop (Sliding Window)
for begin_loc in tqdm(range(0, input_ids.size(1), stride)):
    end_loc = min(begin_loc + seq_len, input_ids.size(1))
    trg_len = end_loc - prev_end_loc 
    
    if end_loc == input_ids.size(1):
        trg_len = end_loc - begin_loc

    # Prepare inputs
    # Note: If device_map="auto" put the model on a specific GPU, we send inputs there.
    # Usually model.device works, but for multi-gpu split it might differ. 
    # Using 'device' variable defined at top (cuda) is usually safe for inputs.
    input_ids_chunk = input_ids[:, begin_loc:end_loc].to(model.device)
    
    target_ids = input_ids_chunk.clone()
    target_ids[:, :-trg_len] = -100 

    with torch.no_grad():
        outputs = model(input_ids_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood * trg_len)

    prev_end_loc = end_loc
    
    # Optional: clean up text to prevent clutter
    # print(f"\033[A\rAverage NLL so far: {torch.stack(nlls).sum() / end_loc:.4f}")
    if end_loc == input_ids.size(1):
        break

# 4. Final Calculation
total_nll = torch.stack(nlls).sum()
total_tokens = end_loc
ppl = math.exp(total_nll / total_tokens)

logger.info(f"Perplexity (WikiText-2): {ppl:.2f}")