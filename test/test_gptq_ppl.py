import argparse
import torch
import math
from datasets import load_dataset
from gptqmodel import GPTQModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Evaluate GPTQ Model Perplexity on WikiText-2")
parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="/home/qingtaoli/models/tman/Qwen-8B-w2g64-gptq",
    help="Path to the GPTQ model directory",
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

# 1. Load Model
# model_path = "/home/qingtaoli/models/tman/Llama-3.1-8B-Instruct-w2g64-gptq"
# model_path = "/home/qingtaoli/models/tman/Qwen-8B-w2g64-gptq"
model_path = args.model_path
model = GPTQModel.load(model_path, device=device)
tokenizer = model.tokenizer

# Ensure the model is in eval mode
model.eval()

# 2. Load and Preprocess Data (The Fix)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Merge all text into one long stream. 
# We join with "\n\n" to preserve document structure, which is standard for WikiText.
logger.info("Tokenizing entire dataset...")
encodings = tokenizer("".join(ds["text"]), return_tensors="pt")

# print(ds["text"])
# exit()

# Define window size. Llama 3 has a huge context, but for PPL evaluation 
# 4096 is a standard, safe context length that balances memory and accuracy.
seq_len = args.seq_len  # How many tokens we process at a time 
stride = args.stride # How many tokens we slide over (smaller = more accurate, slower)

input_ids = encodings.input_ids
max_length = model.config.max_position_embeddings
# Cap seq_len if the model config is smaller (unlikely for Llama 3, but safe)
seq_len = min(seq_len, max_length)

nlls = []
prev_end_loc = 0

logger.info(f"Starting evaluation with Sequence Length: {seq_len} and Stride: {stride}\n")

# 3. Evaluation Loop (Sliding Window)
for begin_loc in tqdm(range(0, input_ids.size(1), stride)):
    end_loc = min(begin_loc + seq_len, input_ids.size(1))
    trg_len = end_loc - prev_end_loc  # How many new tokens we are predicting
    
    # Check if we have reached the end
    if end_loc == input_ids.size(1):
        trg_len = end_loc - begin_loc # Handle the very last chunk strictly

    # Prepare inputs
    input_ids_chunk = input_ids[:, begin_loc:end_loc].to(device)
    
    # We want to predict the 'trg_len' tokens at the end of this chunk.
    # So we mask out the tokens that came before them (the context).
    target_ids = input_ids_chunk.clone()
    target_ids[:, :-trg_len] = -100 # -100 is the standard ignore_index in PyTorch CrossEntropy

    with torch.no_grad():
        # Most HF models compute loss automatically if 'labels' are passed
        outputs = model(input_ids_chunk, labels=target_ids)
        
        # If the model wraps the loss calculation inside (standard HF behavior)
        # The loss returned is the average NLL of the tokens where label != -100
        neg_log_likelihood = outputs.loss

    # Convert back to sum of NLL to accumulate correctly across chunks
    nlls.append(neg_log_likelihood * trg_len)

    prev_end_loc = end_loc
    print(f"\033[A\rAverage NLL so far: {torch.stack(nlls).sum() / end_loc:.4f}")
    if end_loc == input_ids.size(1):
        break

# 4. Final Calculation
total_nll = torch.stack(nlls).sum()
total_tokens = end_loc
ppl = math.exp(total_nll / total_tokens)

logger.info(f"Perplexity (WikiText-2): {ppl:.2f}")