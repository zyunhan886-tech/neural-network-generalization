"""
阶段 2a —— Hidden Layer Size 对照实验。

Research question: 在其它超参固定的情况下，隐藏层宽度如何影响收敛速度与
最终 validation accuracy？是否存在收益递减的拐点？

控制变量：eta、mini_batch_size、epochs、训练集大小、随机种子全部固定，
只变 hidden size。每个配置跑多个 seed，以区分真实差异和随机波动。
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "hidden_size"

# TODO(你来定): 要扫哪些宽度？建议跨一个数量级才能看出拐点，
# 例如 [[10], [30], [100]] 起步，有余力再加 [5] / [200] 或双隐藏层 [30, 30]。
HIDDEN_GRID = [
    # [10],
    # [30],
    # [100],
]

# 控制变量 —— 与 baseline 保持一致
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not HIDDEN_GRID:
        sys.exit("请先在 HIDDEN_GRID 里填入要对比的隐藏层配置。")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    for hidden in HIDDEN_GRID:
        for seed in args.seeds:
            rows = run_trial(
                experiment=EXPERIMENT,
                training_data=training_data,
                evaluation_data=validation_data,
                hidden=hidden,
                eta=ETA,
                mini_batch_size=MINI_BATCH_SIZE,
                epochs=args.epochs,
                seed=seed,
                run_label=f"hidden={'-'.join(map(str, hidden))}",
            )
            append_rows(rows)


if __name__ == "__main__":
    main()
