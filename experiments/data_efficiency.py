"""
阶段 2c —— Training Data Size × Network Capacity 二维实验。

Research question: 训练数据量和模型容量如何交互？具体地：
  1. 准确率随训练样本数如何增长，什么时候开始饱和？
  2. 大网络是否只在数据充足时才体现优势？小数据下大网络是否反而更差（过拟合）？

这是本项目唯一的二维扫描 —— 结论若成立，应能在 "accuracy vs n_train，
按 hidden size 分组" 的折线图上看到不同宽度曲线的交叉。
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "data_efficiency"

# TODO(你来定): 训练集大小按对数尺度取点，覆盖 "明显不够" 到 "全量"。
# 建议 [1000, 5000, 20000, 50000]。
N_TRAIN_GRID = [
    # 1000,
    # 5000,
    # 20000,
    # 50000,
]

# TODO(你来定): 至少两个容量档位才能看出交互效应，建议 [[10], [100]]。
HIDDEN_GRID = [
    # [10],
    # [100],
]

# 控制变量
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not N_TRAIN_GRID or not HIDDEN_GRID:
        sys.exit("请先填入 N_TRAIN_GRID 和 HIDDEN_GRID。")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    # 注意: 这里按固定前缀切片。MNIST 的训练集本身已打乱，所以前 n 条
    # 近似类别均衡；如果想更严谨，可以改成分层抽样并在报告 Limitations 里说明。
    for n_train in N_TRAIN_GRID:
        subset = training_data[:n_train]
        for hidden in HIDDEN_GRID:
            for seed in args.seeds:
                rows = run_trial(
                    experiment=EXPERIMENT,
                    training_data=subset,
                    evaluation_data=validation_data,
                    hidden=hidden,
                    eta=ETA,
                    mini_batch_size=MINI_BATCH_SIZE,
                    epochs=args.epochs,
                    seed=seed,
                    run_label=f"n={n_train},hidden={'-'.join(map(str, hidden))}",
                )
                append_rows(rows)


if __name__ == "__main__":
    main()
