import torch
from safetensors.torch import save_file, load_file
import os
import re


# Paths to the split parts of the model checkpoint
path = "/mnt/sdb1/qingtaoli/Llama-2-7b-bitdistiller/checkpoint-1000"
part1_path = os.path.join(path, "pytorch_model-00001-of-00002.bin")
part2_path = os.path.join(path, "pytorch_model-00002-of-00002.bin")

# Load both parts of the model
checkpoint_part1 = torch.load(part1_path, map_location='cpu')
checkpoint_part2 = torch.load(part2_path, map_location='cpu')
print(f"Loaded model parts from {part1_path} and {part2_path}.")

# Combine the parts into one state_dict
combined_state_dict = {**checkpoint_part1, **checkpoint_part2}

# Save the combined state_dict to a SafeTensor file
safe_tensor_path = os.path.join(path, "model.safetensors")
save_file(combined_state_dict, safe_tensor_path)
safetensor_size = os.path.getsize(safe_tensor_path)

# Load the SafeTensor model to verify
loaded_state_dict = load_file(safe_tensor_path)

# Optionally, load into your model (if you have the model architecture)
# model = YourModel()
# model.load_state_dict(loaded_state_dict)

print(f"Model saved in SafeTensor format at {safe_tensor_path} with size {safetensor_size} bytes.")

pt_index_path = os.path.join(path, "pytorch_model.bin.index.json")
with open(pt_index_path, "r") as f:
    pt_index = f.read()
    print(f"Index file loaded from {pt_index_path}.")

pt_index = re.sub(r'"total_size":\s*\d+', f'"total_size": {safetensor_size}', pt_index)

st_index = pt_index.replace("pytorch_model-00001-of-00002.bin", "model.safetensors")
st_index = st_index.replace("pytorch_model-00002-of-00002.bin", "model.safetensors")
st_index_path = os.path.join(path, "model.safetensors.index.json")
with open(st_index_path, "w") as f:
    f.write(st_index)
    print(f"Index file saved to {st_index_path}.")
