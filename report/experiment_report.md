# Feed-forward Neural Networks on MNIST: A From-Scratch Reproduction and Ablation Study

**Author:** Yunhan
**Date:** <!-- 完成日期 -->

---

## 1. Research Questions

<!--
写成可证伪的问句，不要写成 "研究一下学习率的影响"。例如：
  RQ1 (复现) 一个从零实现的 784-30-10 网络能否复现 Nielsen 报告的 ~95% validation accuracy？
             不同随机种子之间的波动幅度是多少？
  RQ2 (容量) 隐藏层宽度从 10 增到 100，准确率提升是否存在收益递减的拐点？
  RQ3 (学习率) eta 在对数尺度上变化时，训练曲线呈现哪几种定性不同的形态？
  RQ4 (交互) 模型容量的优势是否依赖训练数据量 —— 小数据下大网络是否反而更差？
-->

## 2. Methodology

### 2.1 Implementation
<!-- 从零实现了什么：sigmoid 激活、二次代价函数、反向传播、mini-batch SGD。
     明确说明只依赖 numpy 做矩阵运算，没有用任何深度学习框架。 -->

### 2.2 Data
<!-- MNIST 50,000 train / 10,000 validation / 10,000 test。
     关键一句：所有超参对比都在 validation set 上进行，test set 仅在最终确定配置后使用一次。 -->

### 2.3 Experimental Protocol
<!-- 控制变量表：每个实验变什么、固定什么。
     每个配置跑 3 个随机种子 (0/1/2)，报告均值与标准差。
     固定种子同时控制权重初始化和 mini-batch 洗牌顺序。 -->

| Experiment | Varied | Held constant |
|---|---|---|
| Baseline | — | hidden=[30], eta=3.0, batch=10, 30 epochs |
| Hidden size | <!-- --> | <!-- --> |
| Learning rate | <!-- --> | <!-- --> |
| Data × capacity | <!-- --> | <!-- --> |

## 3. Results

### 3.1 Baseline
<!-- 最终 validation accuracy 的 mean ± std，以及收敛到多少 epoch 开始平台期。
     和书中报告的数字对比。 -->

### 3.2 Hidden Layer Size
<!-- 插图: ![](../results/figures/hidden_size.png)
     配一张表给出每个宽度的 final accuracy 和达到 90% 所需 epoch 数。 -->

### 3.3 Learning Rate
<!-- 插图 + 定性描述三种形态：欠步长、合适、震荡/发散。 -->

### 3.4 Training Data Size × Network Capacity
<!-- 插图 + 重点说明曲线是否交叉。 -->

## 4. Discussion

<!-- 这一节决定报告的分量。不要复述数字，要解释机制：
     - 为什么宽度收益递减？(表达能力不再是瓶颈，瓶颈转移到优化/数据)
     - 为什么 eta 过大会震荡？(在极小值附近来回跨越)
     - 数据量与容量的交互说明了什么？
     - 哪个结果和你事先的预期不符？这是最有价值的一段。 -->

## 5. Limitations

<!-- 诚实列出，这反而加分：
     - 只用了二次代价函数 + sigmoid，没有交叉熵 / ReLU / 正则化
     - 每个配置只有 3 个种子，统计效力有限
     - 没有做超参的联合搜索，各实验是单因素变化，可能错过交互效应
     - data_efficiency 用的是固定前缀切片而非分层抽样
     - 没有早停，固定 30 epochs 可能对不同配置不公平 -->

## 6. Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/baseline.py
python experiments/hidden_size.py
python experiments/learning_rate.py
python experiments/data_efficiency.py

python experiments/plot.py hidden_size
```

原始逐 epoch 记录见 `results/metrics.csv`。

## References

1. M. Nielsen, *Neural Networks and Deep Learning*, 2015. http://neuralnetworksanddeeplearning.com
2. Y. LeCun et al., "Gradient-based learning applied to document recognition," *Proc. IEEE*, 1998.
