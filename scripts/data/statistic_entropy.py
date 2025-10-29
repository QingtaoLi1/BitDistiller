import argparse
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
MODEL_NAME = "/home/qingtaoli/mnt/models/Qwen/Qwen3-14B"
# MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_batch8/checkpoint-600/hf"
MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_base_dense_ckpt/checkpoint-600/hf"
CLIP_PATH = "/home/qingtaoli/models/Qwen/Qwen3-14B/int2-g64-code_nemotron.pt"

# 你的本地数据集文件路径
# DATASET_FILE_PATH = "/home/qingtaoli/data/nemotron-sft-code_500K_block_0_repeat4.jsonl"

# 从数据集中选取前 k 个样本
K_SAMPLES = 64

# 存储 logits 的输出目录
OUTPUT_DIR = "entropy_aime24"

# 并行处理配置
NUM_SAVER_PROCESSES = 4  # None 表示自动设置为 CPU 核心数的一半
BATCH_SIZE = 2             # 用于未来批处理扩展

# --- 2. 自定义数据加载 ---

def load_my_dataset_from_jsonl(file_path: str) -> list[str]:
    texts = []
    try:
        counter = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                if i < 76000:
                    continue
                if line.strip(): # 确保行不为空
                    line_data = json.loads(line)
                    # 假设我们需要的文本在 'text' 键下
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

            # get the token text from softmax
            # tokens = softmax.argmax(dim=-1)
            # token_text = self.tokenizer.convert_ids_to_tokens(tokens[0])
            # print every 10 per line
            # for i in range(3250, 3350, 10):
            #     print(f"{i}\t:{token_text[i:i+10]}")
            # token_text = self.tokenizer.batch_decode(tokens)[0]
            # print(token_text)
            # exit()

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
                if self.args.softmax:
                    all_softmax.append(softmax.detach().cpu())
                if self.args.entropy:
                    all_entropy.append(entropy.detach().cpu())
                del softmax, entropy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # with torch.no_grad():
            #     generated_ids = model.generate(
            #         **inputs,
            #         max_new_tokens=256,   # adjust as needed
            #         do_sample=False       # or True with temperature/top_p if you want randomness
            #     )
            # decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # print(decoded_output)
            # exit()

        # all_entropy_tensor = torch.cat(all_entropy) # (total_samples, seq_len)
        return all_entropy, all_softmax

    @staticmethod
    def _plot_index_log(x, title="Dist. Entropy", xlabel=None, ylabel="log(teacher_entropy)", savepath=None, log: bool = False):
        """
        Plot index vs log(x) for a 1-D numpy array.
        Non-positive values (<= 0) are ignored.
        """
        logger.info(f"Plotting log values, input shape: {x.shape}, savepath: {savepath}")
        x = np.asarray(x)
        # print(x)
        if x.ndim != 1:
            raise ValueError("Input must be 1-D.")
        if log:
            # Keep only positive values (log defined only for > 0)
            mask = x > 0
        else:
            mask = np.ones_like(x, dtype=bool)
        if not np.any(mask):
            raise ValueError("No positive values to take log of.")
        indices = np.arange(len(x))[mask]
        values = x[mask]
        fig, ax = plt.subplots(figsize=(8, 6))
        # ax.plot(indices, np.log(values), marker='o', linestyle='-', markersize=1, linewidth=1)
        if log:
            values = np.log(values)
        ax.plot(indices, values, marker='o', linestyle='-', markersize=1, linewidth=1)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax.minorticks_on()
        ax.grid(True, which='both', linewidth=0.5)

        # plt.figure(figsize=(8, 4.5))
        # plt.plot(indices, np.log(values), marker='o', linestyle='-', markersize=1, linewidth=1)
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.grid(True, which='both', linewidth=0.5)
        # skipped = np.count_nonzero(~mask)
        # if skipped > 0:
        #     plt.text(0.99, 0.01, f"Skipped {skipped} non-positive value(s)",
        #             transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=9)
        # plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def _plot_index_softmax(x, title="Dist. Softmax", xlabel=None, ylabel="softmax(teacher_logits)", savepath=None):
        """
        Plot index vs softmax(x) for a 1-D numpy array.
        Non-positive values (<= 0) are ignored.
        """
        x = np.asarray(x)
        # print(x)
        if x.ndim != 1:
            raise ValueError("Input must be 1-D.")
        # Keep only positive values (log defined only for > 0)
        mask = x > 0
        if not np.any(mask):
            raise ValueError("No positive values to take log of.")
        indices = np.arange(len(x))[mask]
        values = x[mask]
        plt.figure(figsize=(8, 4.5))
        plt.plot(indices, values, marker='o', linestyle='-', markersize=1.5, linewidth=1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        skipped = np.count_nonzero(~mask)
        if skipped > 0:
            plt.text(0.99, 0.01, f"Skipped {skipped} non-positive value(s)",
                    transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=9)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
            print(f"Saved plot to: {savepath}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def _plot_diff_and_origin(diff, teacher, student, title="Dist. Entropy Diff and Origin", xlabel=None, ylabel="Entropy or Diff Value", savepath=None):
        """
        Plot index vs log(x) for a 1-D numpy array.
        Non-positive values (<= 0) are ignored.
        """
        # logger.info(f"Plotting values, input shape: {diff.shape}, savepath: {savepath}")
        indices = np.arange(len(diff))
        if diff.ndim != 1:
            raise ValueError("Input must be 1-D.")
        diff = np.asarray(diff)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.title.set_text(title)
        # ax.plot(indices, np.log(values), marker='o', linestyle='-', markersize=1, linewidth=1)
        ax.plot(indices, diff, marker='o', linestyle='none', markersize=1, linewidth=1, color='black', label='Diff')
        ax.plot(indices, student, marker='o', linestyle='none', markersize=1, linewidth=1, color='red', label='Student')
        ax.plot(indices, teacher, marker='o', linestyle='none', markersize=1, linewidth=1, color='green', label='Teacher')
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax.minorticks_on()
        ax.grid(True, which='both', linewidth=0.5)
        ax.legend()
        if savepath:
            fig.savefig(savepath, dpi=150)
        else:
            plt.show()
        plt.close()

    # --- 5. 工作流程 ---

    def process_bitdistiller_model(self):
        logger.info(f"\n--- 开始处理模型 (bitdistiller): {self.model_path} ---")
        output_dir_model = os.path.join(OUTPUT_DIR, self.model_path.replace("/", "_"))
        os.makedirs(output_dir_model, exist_ok=True)
        logger.info(f"Logits 将被存储在: {output_dir_model}")

        # model = torch.compile(model, mode="max-autotune")
        entropy, softmax = self._process_entropy()
        if self.args.softmax:
            assert len(softmax) > 0, "Softmax list is empty but args.softmax is True."
            softmax_np = np.empty(len(softmax), dtype=object)
            for i, s in enumerate(softmax):
                softmax_np[i] = s.float().cpu().numpy()
            np.save(os.path.join(output_dir_model, "softmax.npy"), softmax_np)
            ### To load: softmax_loaded = np.load("softmax.npy", allow_pickle=True)
            for i in range(softmax_np.shape[0]):
                s = softmax_np[i][0]
                self._plot_index_softmax(
                    np.sort(s, axis=0)[-1::-1],
                    title=f"Dist. Softmax: Sample {i}",
                    xlabel="Token Index",
                    ylabel="Softmax",
                    savepath=os.path.join(output_dir_model, f"softmax_sample_{i}.png")
                )

        if self.args.entropy:
            assert len(entropy) > 0, "Entropy list is empty but args.entropy is True."
            entropy_np = np.empty(len(entropy), dtype=object)
            for i, t in enumerate(entropy):
                entropy_np[i] = t.float().cpu().numpy()
            np.save(os.path.join(output_dir_model, "entropy.npy"), entropy_np)
            for i in range(entropy_np.shape[0]):
                t = entropy_np[i][0]
                self._plot_index_log(
                    np.sort(t, axis=0)[-1::-1],
                    title=f"Dist. Entropy: Sample {i}",
                    xlabel="Token Index",
                    ylabel="log(Entropy)",
                    savepath=os.path.join(output_dir_model, f"entropy_sample_{i}.png")
                )

        logger.info(f"--- 模型 {self.model_path} 处理完毕 ---")

    def process_bf16_model(self):
        origin_mode = False
        if self.q_config is None:
            origin_mode = True

        logger.info(f"\n--- 开始处理模型 ({'bf16' if origin_mode else 'init'}): {self.model_path} ---")
        output_name = os.path.basename(self.model_path) if origin_mode else os.path.basename(self.model_path) + "_init"
        output_dir_model = os.path.join(OUTPUT_DIR, output_name)
        os.makedirs(output_dir_model, exist_ok=True)
        logger.info(f"Logits 将被存储在: {output_dir_model}")

        entropy, softmax = self._process_entropy()
        if self.args.softmax:
            assert len(softmax) > 0, "Softmax list is empty but args.softmax is True."
            softmax_np = np.empty(len(softmax), dtype=object)
            for i, s in enumerate(softmax):
                softmax_np[i] = s.float().cpu().numpy()
            np.save(os.path.join(output_dir_model, "softmax.npy"), softmax_np)
            ### To load: softmax_loaded = np.load("softmax.npy", allow_pickle=True)
            for i in range(softmax_np.shape[0]):
                s = softmax_np[i][0]
                self._plot_index_softmax(
                    np.sort(s, axis=0)[-1::-1],
                    title=f"Dist. Softmax: Sample {i}",
                    xlabel="Token Index",
                    ylabel="Softmax",
                    savepath=os.path.join(output_dir_model, f"softmax_sample_{i}.png")
                )
        if self.args.entropy:
            assert len(entropy) > 0, "Entropy list is empty but args.entropy is True."
            entropy_np = np.empty(len(entropy), dtype=object)
            for i, t in enumerate(entropy):
                entropy_np[i] = t.float().cpu().numpy()
            np.save(os.path.join(output_dir_model, "entropy.npy"), entropy_np)
            for i in range(entropy_np.shape[0]):
                t = entropy_np[i][0]
                self._plot_index_log(
                    np.sort(t, axis=0)[-1::-1],
                    title=f"Dist. Entropy: Sample {i}",
                    xlabel="Token Index",
                    ylabel="log(Entropy)",
                    savepath=os.path.join(output_dir_model, f"entropy_sample_{i}.png")
                )

        logger.info(f"--- 模型 {self.model_path} 处理完毕 ---")

    @staticmethod
    def process_entropy_diff(teacher_worker: "Worker", student_worker: "Worker"):
        output_name = os.path.basename(student_worker.model_path)
        output_dir_model = os.path.join(OUTPUT_DIR, "_".join(student_worker.model_path.split("/")[-3:]))
        # output_dir_model = os.path.join(OUTPUT_DIR, output_name, "entropy_diff")
        os.makedirs(output_dir_model, exist_ok=True)

        teacher_entropy, _ = teacher_worker._process_entropy()
        student_entropy, _ = student_worker._process_entropy()
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
        for i in range(entropy_diff_np.shape[0]):
            d = entropy_diff_np[i][0]
            t = teacher_entropy_np[i][0]
            s = student_entropy_np[i][0]
            # student_worker._plot_index_log(
            #     np.sort(d, axis=0)[-1::-1],
            #     title=f"Dist. Entropy Diff: Sample {i}",
            #     xlabel="Token Index",
            #     ylabel="Entropy Diff",
            #     savepath=os.path.join(output_dir_model, f"entropy_diff_sample_{i}.png")
            # )
            student_worker._plot_diff_and_origin(
                d, t, s,
                title=f"Dist. Entropy Diff and Origin: Sample {i}",
                xlabel="Token Index",
                ylabel="Entropy or Diff Value",
                savepath=os.path.join(output_dir_model, f"entropy_diff_origin_sample_{i}.png")
            )
        logger.info(f"Entropy diff plots saved to: {output_dir_model}")

# --- 6. 主逻辑 ---

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

    # 选取前 k 个样本
    subset_texts = all_texts[:K_SAMPLES]
    logger.info(f"已加载 {len(all_texts)} 个样本，选取前 {len(subset_texts)} 个进行处理。")

    q_config = None
    if args.model_type in ["init", "bitdistiller", "diff"]:
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": 64,  # whether to use group quantization
        }

    if args.model_type == "bitdistiller":
        worker = Worker(MODEL_NAME_STUDENT, "bitdistiller", q_config, subset_texts, args)
        worker.process_bitdistiller_model()
    elif args.model_type == "diff":
        teacher_worker = Worker(MODEL_NAME,         "bf16",         None,     subset_texts, args)
        student_worker = Worker(MODEL_NAME_STUDENT, "bitdistiller", q_config, subset_texts, args)
        Worker.process_entropy_diff(teacher_worker, student_worker)
    else:
        worker = Worker(MODEL_NAME, args.model_type, q_config, subset_texts, args)
        worker.process_bf16_model()

    logger.info("\nAll Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to merge FSDP sharded checkpoints faster using GPU.")
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="Whether to compute and store softmax.",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        help="Whether to compute and store entropy. This is prior to --softmax.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bitdistiller", "bf16", "init", "diff"],
        default="bitdistiller",
        help="Type of model to process. 'bitdistiller' for quantized model, 'bf16' for original bf16 model, 'init' for quantized model with clipping initialization.",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=600,
        help="Checkpoint number to load for the student model (if applicable).",
    )
    args = parser.parse_args()
    # if args.ckpt cannot parse to a number, raise error
    try:
        args.ckpt = int(args.ckpt)
    except ValueError:
        raise ValueError("Error: --ckpt must be an integer.")
    if args.ckpt == 0:
        MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/checkpoint_0/checkpoint-0/"
    else:
        MODEL_NAME_STUDENT = MODEL_NAME_STUDENT.replace("checkpoint-600", f"checkpoint-{args.ckpt}")
    if args.entropy:
        args.softmax = False
    else:
        args.softmax = True

    main(args)
