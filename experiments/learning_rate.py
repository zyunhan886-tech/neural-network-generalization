"""
阶段 2b —— Learning Rate 对照实验。

Research question: 学习率 eta 如何影响训练曲线的形状？过小时收敛多慢，
过大时在哪个量级开始震荡或发散？最优 eta 落在什么区间？

控制变量：网络结构、mini_batch_size、epochs、训练集大小固定，只变 eta。
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "learning_rate"

# TODO(你来定): 学习率要按对数尺度扫，不要等间距。
# 建议 [0.01, 0.1, 1.0, 3.0, 10.0] —— 两端要够极端，才能同时展示
# "慢到学不动" 和 "大到震荡" 这两种失败模式，这比只扫最优区间更有说服力。
ETA_GRID = [
    # 0.01,
    # 0.1,
    # 1.0,
    # 3.0,
    # 10.0,
]

# 控制变量 —— 与 baseline 保持一致
HIDDEN = [30]
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not ETA_GRID:
        sys.exit("请先在 ETA_GRID 里填入要对比的学习率。")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    for eta in ETA_GRID:
        for seed in args.seeds:
            rows = run_trial(
                experiment=EXPERIMENT,
                training_data=training_data,
                evaluation_data=validation_data,
                hidden=HIDDEN,
                eta=eta,
                mini_batch_size=MINI_BATCH_SIZE,
                epochs=args.epochs,
                seed=seed,
                run_label=f"eta={eta}",
            )
            append_rows(rows)


if __name__ == "__main__":
    main()
