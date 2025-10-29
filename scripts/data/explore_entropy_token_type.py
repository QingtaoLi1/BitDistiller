import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PlotSwitcher:
    """ Draw bar chart for the four groups. Each group has several values, use different colors for each series across different groups. """
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
        data = self.all_data[self.current_index]
        title = data['title']

        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel("Token Type")
        self.ax.set_ylabel("Count")
        xtick_labels = ["+/+", "+/-", "-/+", "-/-"]
        size = len(xtick_labels)
        x = np.arange(size) * 3
        self.ax.set_xticks(x, xtick_labels)

        width = 0.1
        for ckpt, cnt in data.items():
            if ckpt in ['title']:
                continue
        
            idx = int(ckpt)
            if idx > 100:
                idx = 95 + 5 * (ckpt // 100)
            self.ax.bar(x + (idx // 5 - 1) * width, cnt, width, label=f'CKPT {ckpt}')

        self.ax.legend()
        plt.tight_layout()

        # Tell Matplotlib to redraw the canvas
        self.fig.canvas.draw_idle()

        

def process_entropy_diff(args):
    entropy_diff_nps = dict()
    teacher_entropy_nps = dict()
    init_entropy_nps = dict()
    DIR = args.path
    INIT_DIR = os.path.join(DIR, "checkpoint_0_checkpoint-0_")
    for ckpt in args.ckpts:
        if ckpt == 0:
            CKPT_DIR = os.path.join(DIR, "checkpoint_0_checkpoint-0_")
        elif ckpt < 100:
            CKPT_DIR = os.path.join(DIR, f"nemotron_code_cakld_ctx16384_H100_step300repeat4_const_lr_1e-6_base_dense_ckpt_checkpoint-{ckpt}_hf")
        else:
            CKPT_DIR = os.path.join(DIR, f"nemotron_code_cakld_ctx16384_H100_top512_batch8_checkpoint-{ckpt}_hf")
        
        entropy_diff_np = np.load(os.path.join(CKPT_DIR, "entropy_diff.npy"), allow_pickle=True)
        teacher_entropy_np = np.load(os.path.join(CKPT_DIR, "teacher_entropy.npy"), allow_pickle=True)
        init_entropy_np = np.load(os.path.join(INIT_DIR, "entropy_diff.npy"), allow_pickle=True)
        entropy_diff_nps[ckpt] = entropy_diff_np
        teacher_entropy_nps[ckpt] = teacher_entropy_np
        init_entropy_nps[ckpt] = init_entropy_np

    switcher = PlotSwitcher()
    for i in range(entropy_diff_nps[args.ckpts[0]].shape[0]):
        cnts = {
            'title': f"Token Count: Sample {i}"
        }
        for ckpt in args.ckpts:
            d = entropy_diff_nps[ckpt][i][0]
            t = teacher_entropy_nps[ckpt][i][0]
            s = init_entropy_nps[ckpt][i][0]

            p1 = [s >= 0, s < 0]
            p2 = [d >= 0, d < 0]
            cnt = [np.sum(np.logical_and(p1[0], p2[0])), np.sum(np.logical_and(p1[0], p2[1])),
                    np.sum(np.logical_and(p1[1], p2[0])), np.sum(np.logical_and(p1[1], p2[1]))]
            logger.info(f"Checkpoint {ckpt}, Sample {i} After vs Before Training Entropy counts:\n"
                        f"SE-TE          AFTER >= 0  AFTER < 0\n"
                        f"BEFORE > 0   : {cnt[0]:>10} {cnt[1]:>10}\n"
                        f"BEFORE < 0   : {cnt[2]:>10} {cnt[3]:>10}\n")

            cnts[ckpt] = cnt

        switcher.add_data(cnts)
    plt.show()



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
        "--ckpts",
        type=comma_separated_list_ckpts,
        default=[600],
        help="Checkpoint numbers to load for the student model (if applicable). Split by comma if multiple.",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Directory to the entropy files.",
    )
    args = parser.parse_args()

    process_entropy_diff(args)
    logger.info("\nAll done!")
