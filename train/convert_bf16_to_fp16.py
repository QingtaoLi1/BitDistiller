# Path to the original bf16 model (SafeTensors format)
import torch
from safetensors.torch import load_file, save_file
import os
import re

# Load the Safetensor model (you'll need the correct path to the .safetensors file)
path = "/mnt/sdb1/qingtaoli/Llama-2-7b-bitdistiller/checkpoint-1000"
safetensor_model_path = os.path.join(path, 'model.safetensors')

# Load the model weights from the Safetensor file
model_dict = load_file(safetensor_model_path)

# Convert the model weights from bf16 to fp16
# Assuming model_dict contains the weights for the model, you will convert each tensor to fp16
converted_dict = {k: v.to(torch.float16) for k, v in model_dict.items()}

# Save the converted weights to a new Safetensor file
new_safetensor_model_path = os.path.join(path, 'model.safetensors')
save_file(converted_dict, new_safetensor_model_path)
safetensor_size = os.path.getsize(new_safetensor_model_path)

# Load the new SafeTensor model to verify
loaded_state_dict = load_file(new_safetensor_model_path)

print(f"Model saved as fp16 Safetensor file at: {new_safetensor_model_path} with size {safetensor_size} bytes.")

pt_index_path = os.path.join(path, "model.safetensors.index.json")
with open(pt_index_path, "r") as f:
    pt_index = f.read()
    print(f"Index file loaded from {pt_index_path}.")

st_index = re.sub(r'"total_size":\s*\d+', f'"total_size": {safetensor_size}', pt_index)
st_index_path = os.path.join(path, "model.safetensors.index.json")
with open(st_index_path, "w") as f:
    f.write(st_index)
    print(f"Index file saved to {st_index_path}.")
