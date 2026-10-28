# MNIST Neural Network: From-Scratch Implementation & Ablation Study

从零实现（仅依赖 numpy）的前馈神经网络，在 MNIST 上复现并开展一组受控对照实验，
研究**网络容量、学习率、训练数据量**对泛化性能的影响。

> 自学项目。基础网络实现参考 Michael Nielsen 的
> *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)*（MIT License），
> 已移植到 Python 3 并重写；实验设计、指标记录、分析与报告为本人工作。

## What's implemented from scratch

- Sigmoid 激活与其导数
- 二次代价函数及其梯度
- 反向传播（逐层 delta 递推）
- Mini-batch 随机梯度下降
- 可复现的随机种子控制（权重初始化 + mini-batch 洗牌）

无 TensorFlow / PyTorch / Keras —— 只有 numpy 的矩阵运算。

## Results

详见 [`report/experiment_report.md`](report/experiment_report.md)。
逐 epoch 原始数据在 [`results/metrics.csv`](results/metrics.csv)，图表在 `results/figures/`。

## Project structure

```
.
├── src/
│   ├── network.py            # 手写网络：前向、反向传播、SGD
│   ├── mnist_loader.py       # MNIST 加载与预处理
│   └── experiment_utils.py   # trial 执行 + CSV 指标记录
├── experiments/
│   ├── baseline.py           # 阶段 1: 复现基础网络
│   ├── hidden_size.py        # 阶段 2a: 容量对照
│   ├── learning_rate.py      # 阶段 2b: 学习率对照
│   ├── data_efficiency.py    # 阶段 2c: 数据量 × 容量
│   └── plot.py               # 阶段 3: 出图
├── results/
│   ├── metrics.csv           # tidy 格式，每行 = 一个 run 的一个 epoch
│   └── figures/
├── report/
│   └── experiment_report.md
├── data/mnist.pkl.gz
└── reference/                # Nielsen 原版 Python 2 代码与书中插图，仅作参考
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python experiments/baseline.py                 # ~1.7s/epoch
python experiments/baseline.py --epochs 10 --seeds 0   # 快速试跑
python experiments/plot.py baseline
```

每个实验脚本重跑时会自动清除自己在 `metrics.csv` 中的旧记录，不会重复累积。

## Methodology notes

- 所有超参对比在 **validation set** 上进行；test set 保留到最终配置确定后才使用一次。
- 每个配置跑多个随机种子，报告均值与离散度，以区分真实效应与随机波动。
- 各实验为单因素受控变化，其余超参与 baseline 保持一致。

## License

实验代码 MIT。`reference/` 下的原始代码版权归 Michael Nielsen (MIT License)。
