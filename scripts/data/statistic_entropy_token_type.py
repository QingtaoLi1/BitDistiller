import argparse
import json
import logging
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

import sys
sys.path.append("../../")
sys.path.append("../../test")
sys.path.append("../../quantization")
from test_utils import pseudo_quantize_model_weight
from qlinear import convertModelToQuant
from clip_utils import apply_clip


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 参数配置 (请根据你的需求修改) ---

# Hugging Face Hub 上的模型名称
# MODEL_NAME = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_token_curriculum_te_0.001-0.1_batch8/checkpoint-100/hf/"
MODEL_NAME = "/home/qingtaoli/models/Qwen/Qwen3-14B"
MODEL_NAME_INIT = "/home/qingtaoli/checkpoints/Qwen/Qwen3-14B/checkpoint_0/checkpoint-0/"
# MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_batch8/checkpoint-600/hf"
# MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_base_dense_ckpt/checkpoint-600/hf"
# MODEL_NAME_STUDENT = "/home/qingtaoli/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_base_dense_ckpt/checkpoint-600/hf"
MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_se_gt_te_warmup50/checkpoint-600/hf"
CLIP_PATH = "/home/qingtaoli/models/Qwen/Qwen3-14B/int2-g64-code_nemotron.pt"

# 你的本地数据集文件路径
# DATASET_FILE_PATH = "/home/qingtaoli/data/nemotron-sft-code_500K_block_0_repeat4.jsonl"

# 从数据集中选取前 k 个样本
K_SAMPLES = 4

# 存储 logits 的输出目录
OUTPUT_DIR = "entropy_aime24_visualize_test"

# # 并行处理配置
# NUM_SAVER_PROCESSES = 4  # None 表示自动设置为 CPU 核心数的一半
# BATCH_SIZE = 2             # 用于未来批处理扩展


def load_my_dataset_from_jsonl(file_path: str) -> list[str]:
    texts = []
    try:
        counter = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                if i < 76000:
                    continue
                if line.strip():
                    line_data = json.loads(line)
                    texts.append(line_data[0][0] + "\n\n<think>" + line_data[0][1])
                    counter += 1
                    if counter >= K_SAMPLES:
                        break
    except FileNotFoundError:
        logger.error(f"错误: 数据文件未找到 at '{file_path}'")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"错误: 解析 JSON 时出错: {e}")
        return []
    except KeyError:
        logger.error(f"错误: 在 JSON 行中未找到 'text' 键。请检查您的数据格式或修改 load_my_dataset_from_jsonl 函数。")
        return []
    return texts


def load_aime24() -> list[str]:
    from datasets import load_dataset
    texts = []
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    for example in dataset:
        texts.append(example["problem"] + "\n\n<think>" + example["solution"])
    return texts


# --- 4. 工作逻辑 ---

class Worker:
    def __init__(self, model_path: str, model_mode: str, q_config: Optional[dict], subset_texts: list[str], args):
        self.model_path: str = model_path
        self.model_mode: str = model_mode   # "bitdistiller", "bf16"
        self.q_config: dict = q_config
        self.subset_texts: list[str] = subset_texts
        self.args = args
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing model: {model_path}, Quantization config: {self.q_config}.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        # self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model_mode == "bitdistiller":
            logger.info(f"Quantizing model weights with config: quant_type=int, bits=2, q_config={self.q_config}")
            pseudo_quantize_model_weight(
                self.model, w_bit=2, q_config=self.q_config, quant_type="int"
            )
        elif self.model_mode == "bf16":
            if self.q_config is not None:
                logger.info("Converting the model to qat, this may take a while...")
                model, _ = convertModelToQuant(self.model, compute_dtype=torch.bfloat16, quant_type="int2-asym", q_group_size=self.q_config["q_group_size"])
                logger.info(f"Loading pre-computed Clipping results from {CLIP_PATH}")
                clip_results = torch.load(CLIP_PATH)
                apply_clip(model, clip_results)
                logger.info("Clipping init successfully!")
        logger.info("Model initialized successfully!")


    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _get_softmax_entropy(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.detach()
            softmax = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(softmax * torch.log_softmax(logits, dim=-1), dim=-1) # (batch_size, seq_len)
        return softmax, entropy

    def _process_entropy(self):
        all_entropy = []
        all_softmax = []
        for i, text in enumerate(tqdm(self.subset_texts, desc=f"处理模型 {self.model_path}")):
            if not text:
                logger.warning(f"警告: 样本 {i} 为空, 已跳过。")
                continue

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=16384)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                softmax, entropy = self._get_softmax_entropy(inputs)
                all_entropy.append(entropy.detach().cpu())
                all_softmax.append(softmax.detach().cpu())
                del softmax, entropy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return all_entropy, all_softmax


    @staticmethod
    def process_entropy_diff(teacher_worker: "Worker", student_worker: "Worker"):
        output_name = os.path.basename(student_worker.model_path)
        output_dir_model = os.path.join(OUTPUT_DIR, "_".join(student_worker.model_path.split("/")[-3:]))
        # output_dir_model = os.path.join(OUTPUT_DIR, output_name, "entropy_diff")
        os.makedirs(output_dir_model, exist_ok=True)

        teacher_entropy, teacher_softmax = teacher_worker._process_entropy()
        student_entropy, student_softmax = student_worker._process_entropy()
        assert len(teacher_entropy) == len(student_entropy), "Teacher and student entropy lists must have the same length."

        entropy_diff = []
        for t, s in zip(teacher_entropy, student_entropy):
            entropy_diff.append(s - t)
        entropy_diff_np = np.empty(len(entropy_diff), dtype=object)
        teacher_entropy_np = np.empty(len(teacher_entropy), dtype=object)
        student_entropy_np = np.empty(len(student_entropy), dtype=object)
        for i, (d, t, s) in enumerate(zip(entropy_diff, teacher_entropy, student_entropy)):
            entropy_diff_np[i] = d.float().cpu().numpy()
            teacher_entropy_np[i] = t.float().cpu().numpy()
            student_entropy_np[i] = s.float().cpu().numpy()

        np.save(os.path.join(output_dir_model, "entropy_diff.npy"), entropy_diff_np)
        np.save(os.path.join(output_dir_model, "teacher_entropy.npy"), teacher_entropy_np)
        np.save(os.path.join(output_dir_model, "student_entropy.npy"), student_entropy_np)
        
        # ### Load ckpt-0 as init entropy
        # init_entropy_np = np.load(os.path.join(os.path.join(OUTPUT_DIR, "_".join(MODEL_NAME_INIT.split("/")[-3:])), "student_entropy.npy"), allow_pickle=True)
        # init_entropy = [torch.tensor(arr) for arr in init_entropy_np]

        ### For each token position, print the token text of: 1) input, 2) teacher max prob, 3) student max prob
        mode = student_worker.args.mode
        tokenizer = teacher_worker.tokenizer
        counts = [0 for _ in range(16)]
        for i, (text, te, ts, se, ss) in tqdm(enumerate(zip(student_worker.subset_texts, teacher_entropy, teacher_softmax, student_entropy, student_softmax))):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=16384)
            input_ids = inputs["input_ids"][0]
            tokens = input_ids.cpu().numpy()
            token_texts = tokenizer.convert_ids_to_tokens(tokens)
            teacher_token_ids = torch.argmax(ts, dim=-1)[0]
            student_token_ids = torch.argmax(ss, dim=-1)[0]
            teacher_token_texts = tokenizer.convert_ids_to_tokens(teacher_token_ids.cpu().numpy())
            student_token_texts = tokenizer.convert_ids_to_tokens(student_token_ids.cpu().numpy())
            with open(os.path.join(output_dir_model, f"token_texts_sample_{i}.txt"), "w", encoding="utf-8") as f:
                # Settings
                group_size = 8
                col_width = 15
                idx_width = 15  # Width for the index column

                # ANSI Color codes for terminal output
                COLOR_GREEN = "\033[92m"       # Green
                COLOR_RED = "\033[91m"         # Red
                COLOR_RESET = "\033[0m"        # Reset color

                # Helper to sanitize tokens so newlines don't break the file layout
                def clean_token(t):
                    return str(t).replace('\n', '\\n').replace('\r', '\\r')

                # Iterate through tokens in chunks of 8
                # set color as follows:
                #   1. Green, (se-te)>0;
                #   2. Red, (se-te)<=0;
                print(f"\n{token_texts[0]}", end='')
                total_len = len(token_texts)
                for start_idx in range(0, total_len, group_size):
                    end_idx = min(start_idx + group_size, total_len)
                    chunk_input = token_texts[start_idx + 1 : min(start_idx + group_size + 1, total_len)]  # +1 to align with next-token prediction
                    chunk_teacher = teacher_token_texts[start_idx:end_idx]
                    chunk_student = student_token_texts[start_idx:end_idx]

                    idx_str = f"{start_idx:<{idx_width}}"
                    if mode == "print":
                        line_input = line_teach = line_stud = ""
                    elif mode == "save":
                        line_input = f"Ground - {start_idx:<{idx_width-9}}"
                        line_teach = f"{'Teacher':<{idx_width}}"
                        line_stud = f"{'Student':<{idx_width}}"
                    for j, (t0, t1, t2) in enumerate(zip(chunk_input, chunk_teacher, chunk_student)):
                        te_val = te[0][start_idx + j].item()
                        se_val = se[0][start_idx + j].item()

                        if (se_val - te_val) > 0:
                            color_mode = COLOR_GREEN
                            if t1 == t2:
                                counts[0] += 1
                            if t0 == t1:
                                counts[2] += 1
                            if t0 == t2:
                                counts[4] += 1
                            counts[6] += 1
                        elif (se_val - te_val) <= 0:
                            color_mode = COLOR_RED
                            if t1 == t2:
                                counts[1] += 1
                            if t0 == t1:
                                counts[3] += 1
                            if t0 == t2:
                                counts[5] += 1
                            counts[7] += 1

                        if mode == "print":
                            t0 = clean_token(t0).replace('Ġ', ' ')
                            t1 = clean_token(t1).replace('Ġ', ' ')
                            t2 = clean_token(t2).replace('Ġ', ' ')
                            line_input += f"{color_mode}{t0}{COLOR_RESET}"
                            line_teach += f"{color_mode}{t1}{COLOR_RESET}"
                            line_stud += f"{color_mode}{t2}{COLOR_RESET}"
                        elif mode == "save":
                            line_input += f"{color_mode}{t0:>{col_width}}{COLOR_RESET}"
                            line_teach += f"{color_mode}{t1:>{col_width}}{COLOR_RESET}"
                            line_stud += f"{color_mode}{t2:>{col_width}}{COLOR_RESET}"
                    if mode == "print":
                        print(line_input, end='')
                    elif mode == "save":
                        f.write(line_input + "\n")
                        f.write(line_teach + "\n")
                        f.write(line_stud + "\n")
                        f.write("\n")
        if mode == "save":
            logger.info(f"Token texts saved to: {output_dir_model}")
        print("Counts of different cases:")
        print(f"                   +    -    ")
        print(f"student=teacher: {counts[0]:<5} {counts[1]:<5}")
        print(f"teacher=ground : {counts[2]:<5} {counts[3]:<5}")
        print(f"student=ground : {counts[4]:<5} {counts[5]:<5}")
        print(f"     Total     : {counts[6]:<5} {counts[7]:<5}")


def main(args):
    """主执行函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用的设备是: {device}")

    # logger.info(f"\n正在从 '{DATASET_FILE_PATH}' 加载数据集...")
    # all_texts = load_my_dataset_from_jsonl(DATASET_FILE_PATH)
    logger.info(f"\n正在从 'AIME24' 加载数据集...")
    all_texts = load_aime24()
    
    if not all_texts:
        logger.error("数据加载失败或数据为空，程序退出。")
        return

    subset_texts = all_texts[:K_SAMPLES]
    logger.info(f"已加载 {len(all_texts)} 个样本，选取前 {len(subset_texts)} 个进行处理。")

    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": 64,  # whether to use group quantization
    }

    teacher_worker = Worker(MODEL_NAME,         "bf16",         None,     subset_texts, args)
    student_worker = Worker(MODEL_NAME_STUDENT, "bitdistiller", q_config, subset_texts, args)
    Worker.process_entropy_diff(teacher_worker, student_worker)

    logger.info("\nAll Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to merge FSDP sharded checkpoints faster using GPU.")
    parser.add_argument(
        "--ckpt",
        type=int,
        default=600,
        help="Checkpoint number to load for the student model (if applicable).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["print", "save"],
        default="print",
        help="Mode of operation: 'print' to display token texts, 'save' to save them to files.",
    )
    args = parser.parse_args()
    # if args.ckpt cannot parse to a number, raise error
    try:
        args.ckpt = int(args.ckpt)
    except ValueError:
        raise ValueError("Error: --ckpt must be an integer.")
    if args.ckpt == 0:
        MODEL_NAME_STUDENT = MODEL_NAME_INIT
    else:
        MODEL_NAME_STUDENT = MODEL_NAME_STUDENT.replace("checkpoint-600", f"checkpoint-{args.ckpt}")

    main(args)
