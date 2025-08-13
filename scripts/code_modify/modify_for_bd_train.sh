sed -i '/if self.dtype == torch.float16:/{
N
s/if self.dtype == torch.float16:\n[[:space:]]*self.check_overflow()/\
        if self.dtype in [torch.float16, torch.bfloat16]:\
            if (self.overflow):\
                print(f"Rank {dist.get_rank()} overflow.")\
            self.check_overflow()/
}' ../../venv_bd/lib/python3.11/site-packages/deepspeed/runtime/zero/stage_1_and_2.py

### This is for using SequentialSampler instead of RandomSampler, in "def _get_train_sampler"
sed -i 's/return RandomSampler(self.train_dataset)/return SequentialSampler(self.train_dataset)/' ../../venv_bd/lib/python3.11/site-packages/transformers/trainer.py

### This is for saving checkpoint-0.
# sed -i '/if args.eval_on_start:/{
# N
# /.*_evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)/a\
#             self._save_checkpoint(model, trial)
# }' ../../venv_bd/lib/python3.11/site-packages/transformers/trainer.py
