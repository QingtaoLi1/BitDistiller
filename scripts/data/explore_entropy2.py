import argparse
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from typing import Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 参数配置 (请根据你的需求修改) ---

# Hugging Face Hub 上的模型名称
# MODEL_NAME = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_token_curriculum_te_0.001-0.1_batch8/checkpoint-100/hf/"
MODEL_NAME = "/home/qingtaoli/mnt/models/Qwen/Qwen3-14B"
MODEL_NAME_STUDENT = "/home/qingtaoli/mnt/checkpoints/Qwen/Qwen3-14B/nemotron_code_cakld_ctx16384_H100_top512_token_curriculum_te_0.001-0.1_batch8/checkpoint-100/hf/"
CLIP_PATH = "/home/qingtaoli/models/Qwen/Qwen3-14B/int2-g64-code_nemotron.pt"

# 你的本地数据集文件路径
DATASET_FILE_PATH = "/home/qingtaoli/data/nemotron-sft-code_500K_block_0_repeat4.jsonl"

# 从数据集中选取前 k 个样本
K_SAMPLES = 64

# 存储 logits 的输出目录
OUTPUT_DIR = "entropy"

# 并行处理配置
NUM_SAVER_PROCESSES = 4  # None 表示自动设置为 CPU 核心数的一半
BATCH_SIZE = 2             # 用于未来批处理扩展

# --- 4. 工作逻辑 ---
class PlotSwitcher:
    def __init__(self, all_data: list[dict] = None):
        self.setup_figure(all_data)

    def setup_figure(self, all_data: list[dict]):
        self.all_data = all_data if all_data else []
        self.current_index = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        # Connect the keyboard event to our function
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        # event.key is 'right', 'left', 'up', 'down', etc.
        if event.key == 'right':
            # Move to the next dataset, wrap around to 0 if at the end
            self.current_index = (self.current_index + 1) % len(self.all_data)
        elif event.key == 'left':
            # Move to the previous dataset, wrap around
            self.current_index = (self.current_index - 1) % len(self.all_data)
        else:
            return
        
        # Redraw the plot with the new data
        self.update_plot()
    
    def add_data(self, data: dict):
        self.all_data.append(data)
    
    def remove_data(self, index: int):
        if 0 <= index < len(self.all_data):
            del self.all_data[index]

    def update_plot(self):
        # Get the data for the current index
        data = self.all_data[self.current_index]
        diff = data['diff']
        teacher = data['teacher']
        student = data['student']
        title = data['title']
        indices = np.arange(len(diff))

        if diff.ndim != 1:
            raise ValueError("Input must be 1-D.")

        avg_ds = []
        avg_ts = []
        avg_ss = []
        for j in range(0, len(diff), 100):
            avg_d = np.mean(diff[j:j+100])
            avg_t = np.mean(teacher[j:j+100])
            avg_s = np.mean(student[j:j+100])
            # print(f"Tokens {j}\t to {j+100}\t: Avg Diff: {avg_d:.4f}, Avg Teacher Entropy: {avg_t:.4f}, Avg Student Entropy: {avg_s:.4f}")
            avg_ds.append(avg_d)
            avg_ts.append(avg_t)
            avg_ss.append(avg_s)
        indices_avg = 100 * np.arange((len(diff)+99)//100) + 50

        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel("Ordered Token Index")
        self.ax.set_ylabel("Entropy or Diff Value")
        self.ax.plot(indices, student, marker='o', linestyle='none', markersize=1, linewidth=1, color='#f08989', label='Student')
        self.ax.plot(indices, teacher, marker='o', linestyle='none', markersize=1, linewidth=1, color="#8ad88a", label='Teacher')
        self.ax.plot(indices, diff, marker='o', linestyle='none', markersize=1, linewidth=1, color='#808080', label='Diff')
        self.ax.plot(indices_avg, avg_ds, marker='o', linestyle='-', markersize=4, linewidth=2, color='black', label='Diff (100-token Avg)')
        self.ax.plot(indices_avg, avg_ss, marker='o', linestyle='-', markersize=4, linewidth=2, color='red', label='Student (100-token Avg)')
        self.ax.plot(indices_avg, avg_ts, marker='o', linestyle='-', markersize=4, linewidth=2, color='green', label='Teacher (100-token Avg)')
        self.ax.axhline(0, color='black', linewidth=1, linestyle='-', zorder=2)
        self.ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        self.ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        self.ax.set_ylim(-3, 12)
        self.ax.minorticks_on()
        self.ax.grid(True, which='both', linewidth=0.5)
        self.ax.legend()

        # Tell Matplotlib to redraw the canvas
        self.fig.canvas.draw_idle()

        

class Worker:
    def __init__(self, model_path: str, model_mode: str, q_config: Optional[dict], subset_texts: list[str], args):
        self.model_path: str = model_path
        self.model_mode: str = model_mode   # "bitdistiller", "bf16"
        self.q_config: dict = q_config
        self.subset_texts: list[str] = subset_texts
        self.args = args


    @staticmethod
    def _plot_diff_and_origin(diff, teacher, student, title="Dist. Entropy Diff and Origin", xlabel=None, ylabel="Entropy or Diff Value", savepath=None):
        """
        Plot index vs log(x) for a 1-D numpy array.
        Non-positive values (<= 0) are ignored.
        """
        # logger.info(f"Plotting values, input shape: {diff.shape}, savepath: {savepath}")
        avg_ds = []
        avg_ts = []
        avg_ss = []
        for j in range(0, len(diff), 100):
            avg_d = np.mean(diff[j:j+100])
            avg_t = np.mean(teacher[j:j+100])
            avg_s = np.mean(student[j:j+100])
            # print(f"Tokens {j}\t to {j+100}\t: Avg Diff: {avg_d:.4f}, Avg Teacher Entropy: {avg_t:.4f}, Avg Student Entropy: {avg_s:.4f}")
            avg_ds.append(avg_d)
            avg_ts.append(avg_t)
            avg_ss.append(avg_s)
        indices_avg = 100 * np.arange((len(diff)+99)//100) + 50

        indices = np.arange(len(diff))
        if diff.ndim != 1:
            raise ValueError("Input must be 1-D.")
        diff = np.asarray(diff)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.title.set_text(title)
        # ax.plot(indices, np.log(values), marker='o', linestyle='-', markersize=1, linewidth=1)
        ax.plot(indices, diff, marker='o', linestyle='none', markersize=1, linewidth=1, color='#808080', label='Diff')
        ax.plot(indices, student, marker='o', linestyle='none', markersize=1, linewidth=1, color='#f08989', label='Student')
        ax.plot(indices, teacher, marker='o', linestyle='none', markersize=1, linewidth=1, color="#8ad88a", label='Teacher')
        # ax.plot(indices_avg, avg_ds, marker='o', linestyle='-', markersize=4, linewidth=2, color='black', label='Diff (100-token Avg)')
        ax.plot(indices_avg, avg_ss, marker='o', linestyle='-', markersize=4, linewidth=2, color='red', label='Student (100-token Avg)')
        ax.plot(indices_avg, avg_ts, marker='o', linestyle='-', markersize=4, linewidth=2, color='green', label='Teacher (100-token Avg)')
        ax.axhline(0, color='black', linewidth=1, linestyle='-', zorder=2)
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

    @staticmethod
    def process_entropy_diff(args, teacher_worker: "Worker", student_worker: "Worker"):
        entropy_diff_nps = dict()
        teacher_entropy_nps = dict()
        student_entropy_nps = dict()
        DIR = "D:\\OneDrive - Microsoft\\Documents\\工作记录\\BitDistiller\\logits\\entropy"
        for ckpt in args.ckpts:
            if ckpt == 0:
                CKPT_DIR = os.path.join(DIR, "checkpoint_0_checkpoint-0_")
            else:
                CKPT_DIR = os.path.join(DIR, f"nemotron_code_cakld_ctx16384_H100_top512_batch8_checkpoint-{ckpt}_hf")
            
            entropy_diff_np = np.load(os.path.join(CKPT_DIR, "entropy_diff.npy"), allow_pickle=True)
            teacher_entropy_np = np.load(os.path.join(CKPT_DIR, "teacher_entropy.npy"), allow_pickle=True)
            student_entropy_np = np.load(os.path.join(CKPT_DIR, "student_entropy.npy"), allow_pickle=True)
            entropy_diff_nps[ckpt] = entropy_diff_np
            teacher_entropy_nps[ckpt] = teacher_entropy_np
            student_entropy_nps[ckpt] = student_entropy_np

        for i in range(entropy_diff_nps[args.ckpts[0]].shape[0]):
            switcher = PlotSwitcher()
            for ckpt in args.ckpts:
                d = entropy_diff_nps[ckpt][i][0]
                t = teacher_entropy_nps[ckpt][i][0]
                s = student_entropy_nps[ckpt][i][0]

                # Sort by d descending, and reorder t and s accordingly
                sorted_indices = np.argsort(t)[-1::-1]
                d = d[sorted_indices]
                t = t[sorted_indices]
                s = s[sorted_indices]
                # print(f"len(d): {len(d)}, len(t): {len(t)}, len(s): {len(s)}")
                # for j in range(3730, 3800):
                #     print(f"Token {j}: Entropy Diff: {d[j]:.4f}, Teacher Entropy: {t[j]:.4f}, Student Entropy: {s[j]:.4f}")

                # student_worker._plot_diff_and_origin(
                #     d, t, s,
                #     title=f"Dist. Entropy Diff and Origin: Sample {i}",
                #     xlabel="Token Index",
                #     ylabel="Entropy or Diff Value",
                #     savepath=None
                # )

                switcher.add_data({
                    'diff': d,
                    'teacher': t,
                    'student': s,
                    'title': f"Dist. Entropy Diff and Origin: Checkpoint {ckpt}, Sample {i}"
                })
            plt.show()
            print(f"Showed plots for Sample {i} across checkpoints.")


# --- 6. 主逻辑 ---

def main(args):
    """主执行函数"""
    if args.model_type == "bitdistiller":
        worker = Worker(MODEL_NAME_STUDENT, "bitdistiller", None, None, args)
        worker.process_bitdistiller_model()
    elif args.model_type == "diff":
        teacher_worker = Worker(MODEL_NAME,         "bf16",         None,     None, args)
        student_worker = Worker(MODEL_NAME_STUDENT, "bitdistiller", None,     None, args)
        Worker.process_entropy_diff(args, teacher_worker, student_worker)
    else:
        worker = Worker(MODEL_NAME, args.model_type, None, None, args)
        worker.process_bf16_model()

    logger.info("\n所有任务已完成！")



def comma_separated_list_ckpts(arg):
    items = arg.split(",")
    for item in items:
        if not item.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid checkpoint: {item}. Must be numeric.")
    if len(items) == 0:
        raise argparse.ArgumentTypeError("At least one checkpoint must be specified.")
    
    # Convert to integers
    return [int(item) for item in items]


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
        "--ckpts",
        type=comma_separated_list_ckpts,
        default=[600],
        help="Checkpoint numbers to load for the student model (if applicable). Split by comma if multiple.",
    )
    args = parser.parse_args()
    if args.entropy:
        args.softmax = False
    else:
        args.softmax = True

    main(args)
