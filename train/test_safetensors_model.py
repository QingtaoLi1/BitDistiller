from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_dir = "/mnt/sdb1/qingtaoli/Phi-3-mini-4k-instruct-bitdistiller-wiki3kalpaca5k_ctx2048/checkpoint-1000/fp16/"
safetensor_file = os.path.join(model_dir, "model.safetensors")
config = AutoConfig.from_pretrained(model_dir)
print(f"Model config:\n{config}")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
state_dict = load_file(safetensor_file)
model = AutoModelForCausalLM.from_config(config)
model.load_state_dict(state_dict)
print("Model loaded successfully.")
model.to(device)

text = "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington. "
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,  # Enable sampling for more diverse outputs
        # top_k=50,        # Limit sampling to the top-k tokens
        # top_p=0.95,      # Use nucleus sampling
        temperature=0.7, # Control randomness
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
