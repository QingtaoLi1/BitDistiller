# Train & Test Pipeline

### BitDistiller Training

See [sing_train_bd_whitney.yaml](scripts/sing/sing_train_bd_whitney.yaml).

To submit singularity job, run:
```shell
amlt run ./scripts/sing/sing_train_bd_whitney.yaml -d <job_description> -y
```

⚠ Note: Only use `-y` if you are sure that no redundant files (e.g. dataset files) will be uploaded to the platform.

For different experiments, modify the `./train.sh xxx` command arguments in the YAML file.

### BitDistiller and OpenR1 Testing

BitDistiller Testing (MMLU, Arc-Challenge, Hellaswag, Piqa, Winogrande) and OpenR1 Testing (AIME'24, GPQA-diamond, Math500, livecodebench).

To sumbit singularity job, run:
```shell
python ./scripts/sing/generate_sing_yaml.py --vc=<vc_target_name> --mode=<tasks_to_test> --model_dir=<path_containing_checkpoints> --ckpts=<checkpoint_steps>
```

An example:
```shell
python ./scripts/sing/generate_sing_yaml.py --vc=msrresrchbasicvc --mode="test_mmlu,test_aime" --model_dir="/mnt/external/checkpoints/Qwen/Qwen3-14B/1b-grad-l3-r0.5_cakld_ctx4096/" --ckpts="400,800,1200,1600"
```

`test_arc` includes 4 tasks: Arc-Challenge, Hellaswag, Piqa, Winogrande. Each checkpoint will consume 1 GPU.

`test_mmlu` includes 1 task: MMLU. Each checkpoint will consume 1 GPU.

`test_openr1` includes 4 tasks: AIME'24, GPQA-diamond, Math500, livecodebench. Each checkpoint and each task will consume 4 GPUs.

For more details, run `python ./scripts/sing/generate_sing_yaml.py --help` or read the source file.


### OpenR1 Training (GRPO)

To be examined. [sing_train_grpo_whitney.yaml](scripts/sing/sing_train_grpo_whitney.yaml)

