# Neural Network Generalization on MNIST: A From-Scratch Study

A feed-forward neural network implemented from scratch in numpy, used to run a
set of controlled experiments on how **network capacity, learning rate, and
training set size** affect generalization on MNIST.

> **Scope and attribution.** This is a self-study project. The core network
> implementation follows Chapter 1 of Michael Nielsen's
> *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)*
> (MIT License); I ported it to Python 3 and rewrote it while working through
> the derivations. The experimental design, metric logging, analysis, and
> report are my own work. The original reference code is kept unmodified in
> `reference/`.

## Implemented from scratch

- Sigmoid activation and its derivative
- Quadratic cost function and its gradient
- Backpropagation (layer-by-layer delta recursion)
- Mini-batch stochastic gradient descent
- Seed control over both weight initialisation and mini-batch shuffling

No TensorFlow, PyTorch, or Keras — only numpy array operations.

## Experiments

| Experiment | Question |
|---|---|
| `baseline` | Does the implementation reproduce the ~95% validation accuracy reported in the book, and how much does it vary across seeds? |
| `hidden_size` | How does hidden layer width affect convergence and final accuracy? Where do the returns diminish? |
| `learning_rate` | How does eta change the shape of the training curve, and where are the failure modes? |
| `data_efficiency` | Do larger networks only pay off with enough data? Does capacity hurt in the low-data regime? |

## Method notes

- All hyper-parameter comparisons are made on the **validation set**. The test
  set is held out and used only once, after the final configuration is chosen.
- Each configuration is run with several random seeds; results are reported as
  a mean with a measure of spread, so that a real effect can be distinguished
  from run-to-run noise.
- Each experiment varies one factor at a time, with all remaining
  hyper-parameters held at their baseline values.

## Results

See [`report/experiment_report.md`](report/experiment_report.md) for the full
write-up. Raw per-epoch measurements are in
[`results/metrics.csv`](results/metrics.csv) and figures in `results/figures/`.

## Project structure

```
.
├── src/
│   ├── network.py            # network: forward pass, backpropagation, SGD
│   ├── mnist_loader.py       # dataset loading and preprocessing
│   └── experiment_utils.py   # trial runner and CSV metric logging
├── experiments/
│   ├── baseline.py           # stage 1: reproduce the baseline network
│   ├── hidden_size.py        # stage 2a: capacity
│   ├── learning_rate.py      # stage 2b: learning rate
│   ├── data_efficiency.py    # stage 2c: data size x capacity
│   └── plot.py               # stage 3: figures
├── results/
│   ├── metrics.csv           # tidy format: one row per epoch per run
│   └── figures/
├── report/
│   └── experiment_report.md
├── data/mnist.pkl.gz
└── reference/                # Nielsen's original Python 2 code and book figures
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python experiments/baseline.py                        # roughly 1.7 s per epoch
python experiments/baseline.py --epochs 10 --seeds 0  # quick check
python experiments/plot.py baseline
```

Re-running an experiment clears its own previous rows from `metrics.csv`, so
repeated runs do not accumulate duplicates.

## License

Experiment code is MIT licensed. The code under `reference/` is
copyright Michael Nielsen, also MIT licensed.
