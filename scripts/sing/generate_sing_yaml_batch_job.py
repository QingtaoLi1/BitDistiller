import os
import argparse
from pathlib import Path
import subprocess


HF_token = ""

sku_mapping = {
    "msroctobasicvc": "NDAMv4",
    "msroctovc": "NDv4",

    # "msrresrchbasicvc": "NDAMv4",
    "msrresrchbasicvc": "NC_A100_v4",
    # "msrresrchbasicvc": "NDH100v5",
    # "msrresrchbasicvc": "NDv4",

    "msrresrchvc": "NC_A100_v4",
    # "msrresrchvc": "NDv4",
}

instance_type_mapping = {
    # A100 40GB
    "NDv4": {
        1: "ND12_v4",
        2: "ND24_v4",
        4: "ND48_v4",
        8: "ND96_v4", # Only NvLink. "ND96rs_v4" or "ND96r_v4" for IB-NvLink.
    },
    # A100 80GB
    "NDAMv4": {
        1: "ND12am_A100_v4",
        2: "ND24am_A100_v4",
        4: "ND48am_A100_v4",
        8: "ND96am_A100_v4", # Only NvLink. "ND96amrs_A100_v4" and "ND96amr_A100_v4" for IB-NvLink.
    },
    "NC_A100_v4": {
        1: "NC24ad_A100_v4",
        2: "NC48ad_A100_v4",
        4: "NC96ad_A100_v4", # Only NvLink. No IB-NvLink.
    },
    # H100 80GB
    "NDH100v5": {
        1: "ND12_H100_v5",   # or "ND12_H100_v5-n1", same below.
        2: "ND24_H100_v5",
        4: "ND48_H100_v5",
        8: "ND96_H100_v5",   # Only NvLink. "ND96r_H100_v5" for IB-NvLink.
    },
}


# YAML-formatted environment setup for different modes
test_arc_mmlu_env = """
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
    - uv pip install vllm==0.8.5.post1 bitsandbytes datasets==2.14.6

    - echo -e "alias ll='ls -al'" >> ~/.bashrc
"""
test_openr1_env = """
    - pwd
    - sudo apt-get update
    - sudo apt-get install -y vim
    - pip install uv
    - uv venv venv_openr1 --python 3.11
    - source venv_openr1/bin/activate
    - uv pip install --upgrade pip setuptools packaging wheel ninja

    - uv pip install vllm==0.8.5.post1 lighteval==0.8.1
    - cd /scratch/amlt_code/test/3rdparty/open-r1
    - GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
    - uv pip install transformers==4.51.3 click==8.2.1
    - uv pip install flash-attn==2.7.4.post1 --no-build-isolation
    - uv pip install datasets==3.6.0

    - echo -e "alias ll='ls -al'" >> ~/.bashrc
"""

def get_test_arc_mmlu_commands(mode, model_info, model_dir, vc):
    if mode == "test_arc":
        py_command = f"python llm_eval.py --model $$CKPT_DIR --eval_tasks arc_challenge,winogrande,hellaswag,piqa --test_set --num_fewshot 0 --bits 2 --group_size 64 --quant_type int"
        log_file_name = "arc.log"
    elif mode == "test_mmlu":
        py_command = f"python llm_eval.py --model $$CKPT_DIR --eval_tasks hendrycksTest-* --test_set --num_fewshot 5 --bits 2 --group_size 64 --quant_type int"
        log_file_name = "MMLU.log"

    num_gpus = 1

    command = f"""
    name: bd_{mode}_{model_info}_{{ckpt:d}}
    sku: {instance_type_mapping[sku_mapping[vc]][num_gpus]}
    identity: managed
    submit_args:
      env:
        _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/656a79af-6a27-4924-ad92-9221860e3bba/resourceGroups/dca-core/providers/Microsoft.ManagedIdentity/userAssignedIdentities/dca-core-identity"
    command:
        - hf auth login --token {HF_token} --add-to-git-credential
        - cd test/general
        - export CKPT_DIR={model_dir}/checkpoint-{{ckpt}}/hf
        - nohup {py_command} 2>&1 | tee -a $$CKPT_DIR/{log_file_name} &
        - pid=$$!
        - wait $$pid
    tags: ["Debug:False"]
    priority: High
    azml_int: True
"""
    return [command]

def get_test_openr1_commands(mode, model_info, model_dir, vc, only_aime=False, only_gpqa=False, only_math500=False, only_livecodebench=False, only_ifeval=False):
    tasks = ["custom|aime24|0|0", "custom|gpqa:diamond|0|0", "custom|math_500|0|0", "extended|lcb:codegeneration|0|0", "extended|ifeval|0|0"]
    if only_aime:
        tasks = ["custom|aime24|0|0"]
    elif only_gpqa:
        tasks = ["custom|gpqa:diamond|0|0"]
    elif only_math500:
        tasks = ["custom|math_500|0|0"]
    elif only_livecodebench:
        tasks = ["extended|lcb:codegeneration|0|0"]
    elif only_ifeval:
        tasks = ["extended|ifeval|0|0"]

    num_gpus = 2

    commands = []
    for task in tasks:
        if task == "custom|aime24|0|0":
            modify_script_name = "aime"
        elif task == "custom|gpqa:diamond|0|0" or task == "custom|math_500|0|0" or task == "extended|ifeval|0|0":
            modify_script_name = "gpqa"
        elif task == "extended|lcb:codegeneration|0|0":
            modify_script_name = "livecodebench"


        run_idle = f"""
        - cd /mnt/external
        - nohup python keep.py --gpus={num_gpus} --interval=1 >/dev/null 2>&1 &
"""

        command = f"""
    name: bd_{mode}_{model_info}_{{ckpt:d}}
    sku: {instance_type_mapping[sku_mapping[vc]][num_gpus]}
    identity: managed
    submit_args:
      env:
        _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/656a79af-6a27-4924-ad92-9221860e3bba/resourceGroups/dca-core/providers/Microsoft.ManagedIdentity/userAssignedIdentities/dca-core-identity"
    command:
        - source venv_openr1/bin/activate
        - hf auth login --token {HF_token}

        - cd /scratch/amlt_code/scripts/code_modify
        - chmod +x ./modify*.sh
        - ./modify_for_openr1_test_{modify_script_name}.sh

{run_idle if mode == "test_livecodebench" else ""}

        - cd /scratch/amlt_code/test/3rdparty/open-r1
        - export NUM_GPUS={num_gpus}

        - export MODEL_DIR={model_dir}/checkpoint-{{ckpt}}/hf
        - export MODEL_ARGS="pretrained=$$MODEL_DIR,dtype=bfloat16,data_parallel_size=$$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={{{{max_new_tokens:32768,temperature:0.6,top_p:0.95}}}},bits=2,group_size=64,quant_type=int"
        - export OUTPUT_DIR=$$MODEL_DIR/evals
        # - export MODEL_ARGS="pretrained=$$MODEL_DIR,dtype=bfloat16,data_parallel_size=$$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={{{{max_new_tokens:32768,temperature:0.0,top_p:1,top_k:1}}}},bits=2,group_size=64,quant_type=int"
        # - export OUTPUT_DIR=$$MODEL_DIR/evals_greedy
        - VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $$MODEL_ARGS "{task}" --custom-tasks src/open_r1/evaluate.py --use-chat-template --output-dir $$OUTPUT_DIR --save-details
        # - cd /mnt/external
        # - nohup python keep.py --gpus=4 --interval=0.2 >/dev/null 2>&1 &
        # - sleep 100000000
    tags: ["Debug:False"]
    priority: High
    azml_int: True
"""
        commands.append(command)
    return commands

def get_yaml_text(vc, mode_env, jobs_text, ckpts):
    yaml_text = \
f"""description: Simple Amulet job on Singularity

target:
  service: sing
  name: {vc}
  workspace_name: dca-singularity

environment:
  image: amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04
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


search:
  job_template:
{jobs_text}
  type: grid
  max_trials: {100}
  params:
    - name: ckpt
      spec: discrete
      values: [{",".join(ckpts)}]
"""
    return yaml_text


# Prepare argument parsing
valid_modes = {"test_openr1", "test_aime", "test_gpqa", "test_math500", "test_livecodebench", "test_arc", "test_mmlu", "test_ifeval"}
def comma_separated_list_mode(arg):
    items = arg.split(",")
    for item in items:
        if item not in valid_modes:
            raise argparse.ArgumentTypeError(f"Invalid mode: {item}")
    return items

def comma_separated_list_ckpts(arg):
    items = arg.split(",")
    for item in items:
        if not item.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid checkpoint: {item}. Must be numeric.")
    return items

parser = argparse.ArgumentParser(description="Generate Singularity YAML for Amulet job")
parser.add_argument("--vc", type=str, required=True, choices=["msrresrchvc", "msrresrchbasicvc", "msroctovc", "msroctobasicvc"], help="Virtual Cluster name")
parser.add_argument("--mode", type=comma_separated_list_mode, required=True, help="Mode of the job. Valid options: " + ", ".join(valid_modes))
parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model checkpoints")
parser.add_argument("--ckpts", type=comma_separated_list_ckpts, required=True, help="Comma-separated list of checkpoints to use")
args = parser.parse_args()


# Build and execute the YAML configuration
num_ckpts = len(args.ckpts)
if num_ckpts <= 0:
    raise ValueError("No valid checkpoints provided. Please provide a comma-separated list of numeric checkpoints.")

for mode in args.mode:
    mode_env = ""
    if mode in ["test_arc", "test_mmlu"]:
        mode_env = test_arc_mmlu_env
    elif mode in ["test_openr1", "test_aime", "test_gpqa", "test_math500", "test_livecodebench", "test_ifeval"]:
        mode_env = test_openr1_env
    else:
        raise ValueError("Invalid mode specified. Choose from 'test_openr1', 'test_aime', 'test_gpqa', 'test_math500', 'test_livecodebench', 'test_arc', or 'test_mmlu'.")

    def extract_last_three_levels(path):
        path = Path(path.rstrip("/").rstrip("\\"))
        return '_'.join(path.parts[-2:])
    model_info = extract_last_three_levels(args.model_dir)

    mode_job = []
    if mode in ["test_arc", "test_mmlu"]:
        mode_job = get_test_arc_mmlu_commands(mode, model_info, args.model_dir, args.vc)
    elif mode == "test_openr1":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc)
    elif mode == "test_aime":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc, only_aime=True)
    elif mode == "test_gpqa":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc, only_gpqa=True)
    elif mode == "test_math500":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc, only_math500=True)
    elif mode == "test_livecodebench":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc, only_livecodebench=True)
    elif mode == "test_ifeval":
        mode_job = get_test_openr1_commands(mode, model_info, args.model_dir, args.vc, only_ifeval=True)
    else:
        raise ValueError("Invalid mode specified. Choose from 'test_openr1', 'test_aime', 'test_gpqa', 'test_math500', 'test_livecodebench', 'test_arc', or 'test_mmlu'.")

    assert len(mode_job) > 0, "No job commands generated. Please check the mode and checkpoints."
    jobs_text = "\n".join(mode_job)

    vc = args.vc
    yaml_text = get_yaml_text(vc, mode_env, jobs_text, args.ckpts)

    output_file = os.path.join("scripts", "sing", f"singularity_{mode}.yaml")
    with open(output_file, 'w') as f:
        f.write(yaml_text)
    print(f"YAML configuration generated: {output_file}")


    # Run the Singularity command to submit the job
    print("Running Singularity command to submit the job...")
    tries = 2
    while True:
        shell_command = f"amlt run {output_file} bd_{mode}_{model_info}_try{tries} -y -d \"{mode},{model_info}\""
        print("Shell command: ", shell_command)

        try:
            process = subprocess.Popen(shell_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=180)
            break
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print("Error: Singularity command timed out.")
            break
        except Exception as e:
            print(f"Error: Singularity command failed with return code {process.returncode} while running the Singularity command: {type(e).__name__}")
            print("STDOUT:", stdout.strip())
            print("STDERR:", stderr.strip())
            tries += 1
            continue
        
    print("Singularity command executed successfully.")
