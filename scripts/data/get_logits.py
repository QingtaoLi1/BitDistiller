import os
import json
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
from multiprocessing import Process, Queue
import shutil
import subprocess

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

MODEL_1_NAME = "/home/qingtaoli/models/Qwen/Qwen3-14B/V1_16k/checkpoint-900"
MODEL_2_NAME = "/home/qingtaoli/models/Qwen/Qwen3-14B/BF16"
CLIP_PATH = "/home/qingtaoli/models/Qwen/Qwen3-14B/int2-g64-code_nemotron.pt"

DATASET_FILE_PATH = "/home/qingtaoli/data/nemotron-sft-code_500K_block_0_repeat4.jsonl"

K_SAMPLES = 30

OUTPUT_DIR = "logits_output_2"
MOUNT_DIR = "/home/qingtaoli/mnt/logits/Qwen/Qwen3-14B/V1_16k/checkpoint-900/"

# --- 2. 自定义数据加载 ---

def load_my_dataset_from_jsonl(file_path: str) -> list[str]:
    texts = []
    try:
        counter = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): # 确保行不为空
                    line_data = json.loads(line)
                    texts.append(line_data[0][0] + line_data[0][1])
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

# --- 4. 工作逻辑 ---

def process_and_save_logits(model, tokenizer, device, text_subset: list[str], output_dir: str):
    model.to(device)
    model.eval()

    for i, text in enumerate(tqdm(text_subset, desc=f"处理模型 {model.config.name_or_path}")):
        if not text:
            logger.warning(f"警告: 样本 {i} 为空, 已跳过。")
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=16384)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits_cpu = logits.detach().cpu().to(torch.float32).numpy()
        file_path = os.path.join(output_dir, f"sample_{i}_logits.npy")
        np.save(file_path, logits_cpu)

def saver_worker(queue: Queue, output_dir: str):
    while True:
        item = queue.get()
        if item is None:  # poison pill
            break
        i, logits_cpu = item
        file_path = os.path.join(output_dir, f"sample_{i}_logits.pt")
        torch.save(logits_cpu, file_path)

        # COPY_BATCH = 10
        # if (i+1) % COPY_BATCH == 0:
        #     logger.info(f"Worker: Moving samples {i+1-COPY_BATCH} to {i+1} to AZURE/logits/Qwen/Qwen3-14B/V1_16k/checkpoint-900/...")
        #     # Copy the saved files, whose indices are from (i-COPY_BATCH+1) to (i), from the saved path to Azure blob path.
        #     # And delete them from the saved path to save space.
        #     # shutil.copytree(output_dir, MOUNT_DIR, dirs_exist_ok=True)

        #     # subprocess.run(["azcopy", "copy", output_dir, '\"https://dcasingularity4556773921.blob.core.windows.net/qingtaoli/logits/Qwen/Qwen3-14B/V1_16k/checkpoint-900/?sv=2023-01-03&spr=https&st=2025-08-25T10%3A19%3A37Z&se=2025-09-01T10%3A19%3A00Z&skoid=0c4a9632-48e2-4f5b-a2ac-e3a97a9a9c9a&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-08-25T10%3A19%3A37Z&ske=2025-09-01T10%3A19%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwlt&sig=sT%2Fivuemjb6cGGDm%2BKZ7vZXSiLmOPv%2BMNuMffrEAHeU%3D\"', "--recursive"])
        #     # os.remove(os.path.join(output_dir, f"sample_*_logits.pt"))
        #     for j in range(i+1-COPY_BATCH, i+1):
        #         src_path = os.path.join(output_dir, f"sample_{j}_logits.pt")
        #         dst_path = os.path.join(MOUNT_DIR, f"sample_{j}_logits.pt")
        #         # Use your preferred method to copy files to Azure
        #         shutil.move(src_path, dst_path)
        #         # os.remove(src_path)

def process_and_save_logits_async(model, tokenizer, device, text_subset: list[str], output_dir: str, batch_size: int = 4):
    model.to(device)
    model.eval()

    queue = Queue(maxsize=3)  # small buffer to avoid OOM
    saver = Process(target=saver_worker, args=(queue, output_dir))
    saver.start()

    for i, text in enumerate(tqdm(text_subset, desc=f"处理模型 {model.config.name_or_path}")):
        if not text:
            logger.warning(f"警告: 样本 {i} 为空, 已跳过。")
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=16384)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits_cpu = logits.detach().cpu()          #.to(torch.bfloat16)
        queue.put((i, logits_cpu))   # enqueue for saving, non-blocking GPU

    queue.put(None)   # send poison pill
    logger.info("等待所有 logits 被保存...")

    # 设置一个时间间隔，每次检查 queue 的长度，并实时更新到 logger.info 中。每次重写输出行，不要写到下一行。
    check_interval = 1  # seconds
    pbar = tqdm(total=queue.qsize(), desc="当前保存队列情况", position=0)
    prev_size = queue.qsize()
    while not queue.empty():
        size = queue.qsize()
        if size < prev_size:
            # 更新进度条
            pbar.update(prev_size - size)
            prev_size = size
        time.sleep(check_interval)

    saver.join()            
    logger.info(f"所有 logits 已成功保存到 '{output_dir}'")

# --- 5. 工作流程 ---

def process_bf16_model(model_path: str, device: torch.device, subset_texts: list[str], q_config: dict = None):
    origin_mode = False
    if q_config is None:
        origin_mode = True

    logger.info(f"\n--- 开始处理模型 ({'bf16' if origin_mode else 'init'}): {model_path} ---")

    output_name = os.path.basename(model_path) if origin_mode else os.path.basename(model_path) + "_init"
    output_dir_model = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(output_dir_model, exist_ok=True)
    logger.info(f"Logits 将被存储在: {output_dir_model}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    model.to(device)

    if not origin_mode:
        logger.info("Converting the model to qat, this may take a while...")
        model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type="int2-asym", q_group_size=q_config["q_group_size"])
        logger.info(f"Loading pre-computed Clipping results from {CLIP_PATH}")
        clip_results = torch.load(CLIP_PATH)
        apply_clip(model, clip_results)
        logger.info("Clipping init successfully!")

    # process_and_save_logits(model, tokenizer, device, subset_texts, output_dir_model)
    process_and_save_logits_async(model, tokenizer, device, subset_texts, output_dir_model, batch_size=2)

    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info(f"--- 模型 {model_path} 处理完毕 ---")


def process_bitdistiller_model(model_path: str, q_config: dict, device: torch.device, subset_texts: list[str]):
    logger.info(f"\n--- 开始处理模型 (bitdistiller): {model_path} ---")

    output_dir_model = OUTPUT_DIR
    os.makedirs(output_dir_model, exist_ok=True)
    logger.info(f"Logits 将被存储在: {output_dir_model}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    model.to(device)

    logger.info(f"Quantizing model weights with config: quant_type=int, bits=2, q_config={q_config}")
    pseudo_quantize_model_weight(
        model, w_bit=2, q_config=q_config, quant_type="int"
    )

    # model = torch.compile(model, mode="max-autotune")
    process_and_save_logits_async(model, tokenizer, device, subset_texts, output_dir_model, batch_size=2)

    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info(f"--- 模型 {model_path} 处理完毕 ---")


# --- 6. 主逻辑 ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用的设备是: {device}")

    logger.info(f"\n正在从 '{DATASET_FILE_PATH}' 加载数据集...")
    all_texts = load_my_dataset_from_jsonl(DATASET_FILE_PATH)
    
    if not all_texts:
        logger.error("数据加载失败或数据为空，程序退出。")
        return

    subset_texts = all_texts[:K_SAMPLES]
    logger.info(f"已加载 {len(all_texts)} 个样本，选取前 {len(subset_texts)} 个进行处理。")

    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": 64,  # whether to use group quantization
    }

    process_bf16_model(MODEL_2_NAME, device, subset_texts)
    process_bf16_model(MODEL_2_NAME, device, subset_texts, q_config)
    process_bitdistiller_model(MODEL_1_NAME, q_config, device, subset_texts)


    logger.info("所有任务已完成！")

if __name__ == '__main__':
    main()