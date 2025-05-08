from gptqmodel import GPTQModel, QuantizeConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_PATH = "/mnt/external/checkpoints/meta-llama/Llama-3.1-8B-Instruct/alpaca5kwiki3k_ctx4096_MI300/checkpoint-1000/"
MODEL_PATH = "/datadisk/checkpoints/Meta-Llama-3.1-8B-Instruct/alpaca5kwiki3k_ctx4096/checkpoint-400/"
SAVE_PATH = os.path.join(MODEL_PATH, "gptqmodel")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)

calibration_dataset = [
    tokenizer(
        "The world is a wonderful place full of beauty and love."
    )
]
quant_config = QuantizeConfig(bits=2, group_size=64, desc_act=False, sym=False)
model = GPTQModel.from_pretrained(MODEL_PATH, quant_config)
model.eval()
model.quantize(calibration_dataset)
model.save_quantized(SAVE_PATH)

model = GPTQModel.from_quantized(SAVE_PATH, device="cuda:0", trust_remote_code=True)
model.eval()
print("Model loaded successfully.")
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)