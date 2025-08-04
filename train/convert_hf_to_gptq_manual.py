import torch
import json
from safetensors.torch import save_file
import numpy as np
import os
import gptqmodel
from gptqmodel import GPTQModel
print(f"Using gptqmodel version: {gptqmodel.__version__}")


def get_hf_quant_params(module, n_bit=2, zero_point=True, q_group_size=64):
    """
    Placeholder function: You need to implement this based on how
    your Hugging Face model stores its w2g64 parameters.

    Args:
        module: The Hugging Face linear layer module.
        group_size: The group size (e.g., 64).

    Returns:
        A tuple (qweight_unpacked, scales, zero_points_unpacked, bias):
        - qweight_unpacked: (out_features, in_features), dtype torch.intX, values 0-3
        - scales: (out_features, in_features // group_size), dtype torch.bfloat16
        - zero_points_unpacked: (out_features, in_features // group_size), dtype torch.intX, values 0-3
        - bias: (out_features,), dtype torch.bfloat16 or None
    """
    org_w_type = module.weight.dtype
    w = module.weight.to(torch.float32)
    bias = module.bias

    org_w_shape = w.shape
    print(f"Original weight shape: {org_w_shape}")
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)

    w = w.reshape(org_w_shape).detach().numpy()
    scales = scales.detach().numpy().reshape(w.shape[0], -1)
    zeros = zeros.detach().numpy().reshape(w.shape[0], -1) if zero_point else None

    def get_output(arr):
        return torch.from_numpy(arr).to(org_w_type)
    
    if zero_point:
        w = w.astype(np.uint8)
        # zeros = (zeros - (2 ** (n_bit - 1))) * scales     # seems need to keep int in GPTQ format
        zeros = zeros.astype(np.int32)
        return get_output(w), get_output(scales).t(), get_output(zeros).t(), bias
    else:
        w = (w - min_int).astype(np.uint8)
        zeros = zeros.astype(np.int32)
        return get_output(w), get_output(scales).t(), get_output(zeros).t(), bias

    # Example dummy return for testing the rest of the logic:
    # out_features, in_features = module.weight.shape
    # dummy_qweight = torch.randint(0, 4, (out_features, in_features), device=module.weight.device)
    # dummy_scales = torch.randn((out_features, in_features // group_size), device=module.weight.device).half()
    # dummy_zp = torch.randint(0, 4, (out_features, in_features // group_size), device=module.weight.device)
    # dummy_bias = module.bias.data if module.bias is not None else None
    # return dummy_qweight, dummy_scales, dummy_zp, dummy_bias


def convert_layer_to_gptq_format(hf_qweight_unpacked, hf_scales, hf_zero_points_unpacked, bias_orig, bits, group_size):
    """
    Converts parameters of a single layer to GPTQ format.
    """
    if bits != 2:
        raise ValueError("This conversion script is designed for bits=2.")

    out_features, in_features = hf_qweight_unpacked.shape
    device = hf_qweight_unpacked.device

    # 1. Orient parameters for GPTQ (weights/scales/zp are typically KxN where K is input dim axis)
    # hf_qweight_unpacked: (out_features, in_features) -> (in_features, out_features)
    qweight_gptq_oriented = hf_qweight_unpacked.T.contiguous()
    # hf_scales: (out_features, in_features // group_size) -> (in_features // group_size, out_features)
    scales_gptq_oriented = hf_scales.contiguous().to(torch.bfloat16)
    # hf_zero_points_unpacked: (out_features, in_features // group_size) -> (in_features // group_size, out_features)
    zeropoints_gptq_oriented = hf_zero_points_unpacked.T.contiguous()

    # Validate zero points are in the correct range [0, 2^bits - 1]
    if not ((zeropoints_gptq_oriented >= 0) & (zeropoints_gptq_oriented < (1 << bits))).all():
        raise ValueError(f"Zero points must be in the range [0, { (1 << bits) -1}] for {bits}-bit quantization.")

    # 2. Pack qweight
    # Input shape: (in_features, out_features), values 0..3
    # Output shape: (in_features // 16, out_features), dtype torch.int32
    pack_factor = 32 // bits
    if in_features % pack_factor != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by pack_factor ({pack_factor})")

    qweight_final = torch.zeros((in_features // pack_factor, out_features), dtype=torch.int32, device=device)
    # print(f"qweight_final shape: {qweight_final.shape}")
    # print(f"qweight_gptq_oriented shape: {qweight_gptq_oriented.shape}")
    for i in range(pack_factor):
        qweight_final |= qweight_gptq_oriented[i::pack_factor, :].to(torch.int32) << (i * bits)

    # 3. Pack qzeros
    # Input shape: (in_features // group_size, out_features), values 0..3
    # Output shape: (in_features // group_size, out_features // 16), dtype torch.int32
    num_groups = in_features // group_size
    if out_features % pack_factor != 0:
        raise ValueError(f"out_features ({out_features}) must be divisible by pack_factor ({pack_factor}) for zero points")

    qzeros_final = torch.zeros((num_groups, out_features // pack_factor), dtype=torch.int32, device=device)
    # Transpose zeropoints_gptq_oriented to (out_features, num_groups) to pack along out_features
    zp_temp_oriented = zeropoints_gptq_oriented.T.contiguous() # Now shape (out_features, num_groups)
    # print(f"zp_temp_oriented shape: {zp_temp_oriented.shape}")
    # print(f"qzeros_final shape: {qzeros_final.shape}")
    for i in range(pack_factor):
        qzeros_final |= zp_temp_oriented[:, i::pack_factor].to(torch.int32) << (i * bits)
    qzeros_final = qzeros_final.contiguous() # Transpose back to (num_groups, out_features // pack_factor)


    # 4. Scales are already correctly oriented and typed
    scales_final = scales_gptq_oriented

    # 5. Bias
    bias_final = bias_orig.data.clone() if bias_orig is not None else None

    # 6. Generate g_idx
    # g_idx maps input feature index to group index.
    # For standard grouping (consecutive columns), it's simply floor(index / group_size).
    g_idx = torch.arange(in_features, dtype=torch.int32, device=device)
    if group_size != -1: # -1 means group size is in_features
        g_idx = g_idx // group_size
    else:
        # If group_size is -1, all features belong to group 0
        g_idx = torch.zeros(in_features, dtype=torch.int32, device=device)

    # Ensure g_idx has the correct shape and values
    assert g_idx.shape == (in_features,)
    assert g_idx.max() < num_groups

    return qweight_final, qzeros_final, scales_final, g_idx, bias_final


def convert_hf_to_gptq(hf_model_path, save_path, bits=2, group_size=64):
    """
    Loads a Hugging Face model, converts its linear layers to GPTQ format,
    and saves the new model.

    Args:
        hf_model_path (str): Path to the Hugging Face model.
        save_path (str): Path to save the GPTQ formatted model.
        bits (int): Number of bits for quantization (must be 2).
        group_size (int): Group size for quantization.
    """
    if bits != 2:
        raise ValueError("This script is specialized for 2-bit conversion.")

    print(f"Loading Hugging Face model from: {hf_model_path}")
    # Load the model in a way that you can access its layers.
    # device_map="auto" might be problematic if layers are replaced by custom code
    # that is not recognized by accelerate. For conversion, CPU might be safer
    # if memory allows, or process layer by layer.
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        # It's crucial to load the model such that your get_hf_quant_params can work.
        # If it has custom code, trust_remote_code=True might be needed.
        # torch_dtype=torch.bfloat16 is usually fine as a host for "fake fp16".
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            config=config,
            torch_dtype=torch.bfloat16, # Or the dtype your 'fake fp16' model uses
            trust_remote_code=True,
            low_cpu_mem_usage=True # If model is large
        )
        model.eval() # Set to eval mode
        model.cpu() # Move to CPU to ensure all params are accessible and to avoid GPU OOM during conversion
        print("Model loaded on CPU for conversion.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your `get_hf_quant_params` function can correctly access parameters from the loaded model.")
        return

    new_state_dict = model.state_dict().copy() # Start with a copy of original state dict
    layer_prefixes_to_convert = [] # Store prefixes of converted layers

    print("Iterating through model layers to find linear layers for conversion...")
    for name, module in model.named_modules():
        # Identify linear layers. Adjust types if your model uses different linear layer classes.
        if isinstance(module, torch.nn.Linear) and (
                "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or
                "gate_proj" in name or "up_proj" in name or "down_proj" in name):
              
            print(f"Processing layer: {name} of type {type(module)}")
            try:
                # This is where you extract your specific parameters
                hf_qweight_unpacked, hf_scales, hf_zp_unpacked, hf_bias = \
                    get_hf_quant_params(module, bits, True, group_size)

                # Perform the conversion to GPTQ format
                g_qweight, g_qzeros, g_scales, g_idx, g_bias = convert_layer_to_gptq_format(
                    hf_qweight_unpacked, hf_scales, hf_zp_unpacked, hf_bias, bits, group_size
                )

                # Replace original weight and bias, add new quant params
                # First, remove the original fp16 weight if it exists in the state_dict
                # (it might not if the layer is already a custom quantized one)
                if f"{name}.weight" in new_state_dict:
                    del new_state_dict[f"{name}.weight"]
                if hf_bias is not None and f"{name}.bias" in new_state_dict: # bias might be kept or replaced
                     del new_state_dict[f"{name}.bias"] # remove old if replacing

                new_state_dict[f"{name}.qweight"] = g_qweight.cpu()
                new_state_dict[f"{name}.qzeros"] = g_qzeros.cpu()
                new_state_dict[f"{name}.scales"] = g_scales.cpu()
                new_state_dict[f"{name}.g_idx"] = g_idx.cpu()
                if g_bias is not None:
                    new_state_dict[f"{name}.bias"] = g_bias.cpu()

                if name not in layer_prefixes_to_convert:
                    layer_prefixes_to_convert.append(name)
                print(f"Successfully converted layer: {name}")

            except NotImplementedError as nie:
                print(f"Skipping layer {name}: `get_hf_quant_params` is not implemented. {nie}")
            # except Exception as e:
            #     print(f"Could not convert layer {name}: {e}. Skipping...")
            #     exit()

    if not layer_prefixes_to_convert:
        print("No layers were converted. Please check your model structure and `get_hf_quant_params`.")
        return

    # Create quantize_config.json
    quantize_config = {
        "bits": bits,
        "group_size": group_size,
        "sym": False,  # Assuming asymmetric due to explicit zero_points.
                      # Change to True if your quantization is symmetric and zero_points are trivial (e.g., all mid-range)
        "desc_act": False, # Activation order. Assuming False for direct conversion.
        "model_file_base_name": "model", # Or the name of your safetensors file without extension
        "quant_method": "gptq",
        "checkpoint_format": "gptq_v2",
        "meta": {
            "quantizer": "gptqmodel:2.2.0"
        },
    }

    os.makedirs(save_path, exist_ok=True)
    quantize_config_path = os.path.join(save_path, "quantize_config.json")
    with open(quantize_config_path, "w") as f:
        json.dump(quantize_config, f, indent=2)
    print(f"Saved quantize_config.json to {quantize_config_path}")

    # Save the new state_dict
    # Ensure all tensors are on CPU before saving if they aren't already
    for k in new_state_dict:
        if isinstance(new_state_dict[k], torch.Tensor):
            new_state_dict[k] = new_state_dict[k].cpu()

    model_save_path = os.path.join(save_path, "model.safetensors") # Or "pytorch_model.bin"
    metadata = {"format": "pt"}
    save_file(new_state_dict, model_save_path, metadata=metadata)
    print(f"Saved GPTQ model weights to {model_save_path}")

    # Save other necessary files like config.json, tokenizer_config.json, etc.
    # The original model's config should be fine, but we need to ensure it knows
    # it's a quantized model now by having `quantization_config` attribute
    # (though HF usually infers from quantize_config.json and qweight presence).
    model.config.quantization_config = quantize_config # Add it to the main config
    model.config.save_pretrained(save_path)
    print(f"Saved model config to {save_path}")

    # If you have a tokenizer, save it too
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        print(f"Saved tokenizer files to {save_path}")
    except Exception as e:
        print(f"Could not save tokenizer: {e}. You may need to copy it manually.")

    print("Conversion to GPTQ format complete.")
    print(f"Converted layers: {layer_prefixes_to_convert}")

    # model = AutoModelForCausalLM.from_pretrained(
    #     save_path,
    #     torch_dtype=torch.float32,
    #     trust_remote_code=True,
    #     device_map="cuda:0",
    # )
    model = GPTQModel.from_quantized(save_path, device="cuda:0", trust_remote_code=True)
    model.cuda()
    model.eval()
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GPTQ format.")
    parser.add_argument('--hf_model_path', type=str, required=True, help='Path to the Hugging Face model directory.')
    args = parser.parse_args()

    # --- YOU NEED TO CONFIGURE THESE ---
    # Path to your Hugging Face "fake fp16" (w2g64) model directory
    HF_MODEL_PATH = args.hf_model_path
    # Path where the GPTQ converted model will be saved
    GPTQ_SAVE_PATH = os.path.join(HF_MODEL_PATH, "gptq_converted")
    BITS = 2
    GROUP_SIZE = 64 # Your w2g64 uses group size 64

    print("Starting conversion process...")
    print("IMPORTANT: You MUST implement the `get_hf_quant_params` function ")
    print("           to correctly extract quantization parameters from your specific model.")

    # Example usage (ensure paths are correct and get_hf_quant_params is implemented):
    convert_hf_to_gptq(HF_MODEL_PATH, GPTQ_SAVE_PATH, BITS, GROUP_SIZE)
    print("Conversion process finished.")



