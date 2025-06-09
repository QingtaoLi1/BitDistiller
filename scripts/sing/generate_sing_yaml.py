import os
import argparse
from pathlib import Path
import subprocess


HF_token = ""


parser = argparse.ArgumentParser(description="Generate Singularity YAML for Amulet job")
parser.add_argument("--vc", type=str, required=True, choices=["msrresrchvc", "msrresrchbasicvc"], help="Virtual Cluster name")
parser.add_argument("--mode", type=str, required=True, choices=["test_openr1", "test_arc", "test_mmlu"], help="Mode of the job")
parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model checkpoints")
parser.add_argument("--ckpts", type=str, required=True, help="Comma-separated list of checkpoints to use")
args = parser.parse_args()

ckpts = args.ckpts.split(",")
ckpts = [ckpt.strip() for ckpt in ckpts if ckpt.strip().isdigit()]
num_ckpts = len(ckpts)
if num_ckpts <= 0:
    raise ValueError("No valid checkpoints provided. Please provide a comma-separated list of numeric checkpoints.")

mode_env = ""
if args.mode in ["test_arc", "test_mmlu"]:
    mode_env = """
    - pwd
    - sudo apt-get update
    - sudo apt-get install -y vim
    - pip install uv
    - uv venv venv_bd --python 3.11
    - source venv_bd/bin/activate
    - uv pip install --upgrade pip setuptools packaging wheel ninja

    - uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    - cd /scratch/amlt_code
    - uv pip install -r ./requirement_20250507_transformers_4.51.3.txt
    - uv pip install flash-attn==2.7.4.post1 --no-build-isolation
    - uv pip install vllm==0.8.5.post1 bitsandbytes

    - echo -e "alias ll='ls -al'" >> ~/.bashrc
"""
elif args.mode in ["test_openr1"]:
    mode_env = """
    - pwd
    - sudo apt-get update
    - sudo apt-get install -y vim
    - pip install uv
    - uv venv venv_openr1 --python 3.11
    - source venv_openr1/bin/activate
    - uv pip install --upgrade pip setuptools packaging wheel ninja

    - uv pip install vllm==0.7.2
    - cd /scratch/amlt_code/test/3rdparty/open-r1
    - GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
    - uv pip install transformers==4.51.3
    - uv pip install flash-attn==2.7.4.post1 --no-build-isolation

    - echo -e "alias ll='ls -al'" >> ~/.bashrc
"""
    mode_job = "mmlu"
else:
    raise ValueError("Invalid mode specified. Choose from 'test_openr1', 'test_arc', or 'test_mmlu'.")

def extract_last_three_levels(path):
    path = Path(path.rstrip("/").rstrip("\\"))
    return '_'.join(path.parts[-2:])
model_info = extract_last_three_levels(args.model_dir)

mode_job = []
if args.mode in ["test_arc", "test_mmlu"]:
    assert num_ckpts <= 4, "For 'test_arc', the number of checkpoints must be 4 or fewer."
    py_command = ""
    log_file_name = ""
    if args.mode == "test_arc":
        py_command = f"python llm_eval.py --model $$CKPT_DIR --eval_tasks arc_challenge,winogrande,hellaswag,piqa --test_set --num_fewshot 0 --bits 2 --group_size 64 --quant_type int"
        log_file_name = "arc.log"
    elif args.mode == "test_mmlu":
        py_command = f"python llm_eval.py --model $$CKPT_DIR --eval_tasks hendrycksTest-* --test_set --num_fewshot 5 --bits 2 --group_size 64 --quant_type int"
        log_file_name = "MMLU.log"

    command = f"""
- name: bd_{args.mode}_{model_info}_{args.ckpts}
  sku: NC_A100_v4:G{num_ckpts if num_ckpts <= 2 else 4}
  identity: managed
  submit_args:
    env:
      _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/656a79af-6a27-4924-ad92-9221860e3bba/resourceGroups/dca-core/providers/Microsoft.ManagedIdentity/userAssignedIdentities/dca-core-identity"
  command:
    - huggingface-cli login --token {HF_token}
    - cd test/general
"""
    for i, ckpt in enumerate(ckpts):
        command += f"""
    - export CKPT_DIR={args.model_dir}/checkpoint-{ckpt}/
    - CUDA_VISIBLE_DEVICES={i} nohup {py_command} > $$CKPT_DIR/{log_file_name} 2>&1 &
    - pid{i}=$$!
"""
    command += """
    - wait """
    for i in range(len(ckpts)):
        command += f"$$pid{i} "
    command += """
  tags: ["Debug:False"]
  priority: High
  azml_int: True
"""
    mode_job = [command]
elif args.mode == "test_openr1":
    for i, ckpt in enumerate(ckpts):
        for task in ["aime24", "gpqa:diamond", "math_500"]:
            command = f"""
- name: bd_{args.mode}_{model_info}_{ckpt}_{task}
  sku: NC_A100_v4:G4
  identity: managed
  submit_args:
    env:
      _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/656a79af-6a27-4924-ad92-9221860e3bba/resourceGroups/dca-core/providers/Microsoft.ManagedIdentity/userAssignedIdentities/dca-core-identity"
  command:
    - source venv_openr1/bin/activate
    - huggingface-cli login --token {HF_token}

    - cd /scratch/amlt_code/scripts/code_modify
    - chmod +x ./modify*.sh
    - ./modify_for_openr1_test_{"aime" if task == "aime24" else "gpqa"}.sh

    - cd /scratch/amlt_code/test/3rdparty/open-r1
    - export NUM_GPUS=4

    - export MODEL_DIR={args.model_dir}/checkpoint-{ckpt}/
    - export MODEL_ARGS="pretrained=$$MODEL_DIR,dtype=bfloat16,data_parallel_size=$$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}},bits=2,group_size=64,quant_type=int"
    - export OUTPUT_DIR=$$MODEL_DIR/evals
    - lighteval vllm $$MODEL_ARGS "custom|{task}|0|0" --custom-tasks src/open_r1/evaluate.py --use-chat-template --output-dir $$OUTPUT_DIR --save-details
  tags: ["Debug:False"]
  priority: High
  azml_int: True
"""
            mode_job.append(command)
else:
    raise ValueError("Invalid mode specified. Choose from 'test_openr1', 'test_arc', or 'test_mmlu'.")

assert len(mode_job) > 0, "No job commands generated. Please check the mode and checkpoints."
jobs_text = "\n".join(mode_job)

vc = args.vc
yaml_text = f"""
description: Simple Amulet job on Singularity

target:
  service: sing
  name: {vc}
  workspace_name: dca-singularity

environment:
  image: amlt-sing/acpt-torch2.5.0-py3.10-cuda12.4-ubuntu22.04
  setup: {mode_env}

storage:
  input:
    storage_account_name: dcasingularity4556773921
    container_name: qingtaoli
    mount_dir: /mnt/input
  output:
    storage_account_name: dcasingularity4556773921
    container_name: qingtaoli
    mount_dir: /mnt/output
  external:
    storage_account_name: dcasingularity4556773921
    container_name: qingtaoli
    mount_dir: /mnt/external

code:
  # local directory of the code. this will be uploaded to the server.
  # $CONFIG_DIR is expanded to the directory of this config file
  local_dir: ./

data:
  storage_id: external


jobs:
{jobs_text}
"""

output_file = os.path.join("scripts", "sing", f"singularity_{args.mode}.yaml")
with open(output_file, 'w') as f:
    f.write(yaml_text)
print(f"YAML configuration generated: {output_file}")
exit()

print("Running Singularity command to submit the job...")
shell_command = f"amlt run {output_file} -y -d {args.mode},{model_info}"
# result = subprocess.run(shell_command, shell=True, capture_output=True, text=True)
# print("STDOUT:", result.stdout)
# print("STDERR:", result.stderr)

process = subprocess.Popen(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
for line in process.stdout:
    print("STDOUT:", line.strip())
for line in process.stderr:
    print("STDERR:", line.strip())
# Wait for process to complete
process.wait()
