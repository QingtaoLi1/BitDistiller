sed -i '/if self.dtype == torch.float16:/{
N
s/if self.dtype == torch.float16:\n[[:space:]]*self.check_overflow()/\
        if self.dtype in [torch.float16, torch.bfloat16]:\
            if (self.overflow):\
                print(f"Rank {dist.get_rank()} overflow.")\
            self.check_overflow()/
}' ../../venv_bd/lib/python3.11/site-packages/deepspeed/runtime/zero/stage_1_and_2.py
