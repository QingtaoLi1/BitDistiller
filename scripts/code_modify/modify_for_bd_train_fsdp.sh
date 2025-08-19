### This is for using SequentialSampler instead of RandomSampler, in "def _get_train_sampler"
sed -i 's/return RandomSampler(self.train_dataset)/return SequentialSampler(self.train_dataset)/' ../../venv_bd/lib/python3.11/site-packages/transformers/trainer.py

