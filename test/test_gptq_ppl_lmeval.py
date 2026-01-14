import argparse
import logging
import json
import lm_eval
from lm_eval import utils as lm_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate GPTQ Model Perplexity using lm_eval")
parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="/home/qingtaoli/models/tman/Qwen-8B-w2g64-gptq",
    help="Path to the GPTQ model directory",
)
parser.add_argument(
    "--batch_size",
    type=str,
    default="auto",
    help="Batch size (e.g., 'auto', '8', '16'). 'auto' usually finds the max fit.",
)
args = parser.parse_args()

def main():
    logger.info(f"Loading model from: {args.model_path}")

    # specific arguments for the model loader
    # trust_remote_code=True is essential for Qwen models
    # parallelize=True helps if you have multiple GPUs
    model_args = f"pretrained={args.model_path},trust_remote_code=True"

    # Evaluate
    # 'wikitext' in lm_eval maps to wikitext-2-raw-v1 by default in recent versions
    logger.info("Starting evaluation on wikitext...")
    results = lm_eval.simple_evaluate(
        model="hf",           # Uses standard HuggingFace loader (works with GPTQ if config is present)
        model_args=model_args,
        tasks=["wikitext"],   # The standard Wikitext-2 perplexity task
        device="cuda",
        batch_size=args.batch_size,
    )

    # Output results
    # lm_eval returns a complex dictionary; we extract the perplexity metric
    if "wikitext" in results["results"]:
        print(results["results"]["wikitext"])
        ppl = results["results"]["wikitext"].get("word_perplexity,none")        
        print("\n" + "="*40)
        print(f"Model: {args.model_path}")
        print(f"Wikitext-2 Perplexity: {ppl}")
        print("="*40)
    else:
        print("Could not find wikitext results.")
        print(json.dumps(results["results"], indent=2))

if __name__ == "__main__":
    main()