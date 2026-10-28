# Neural Network Generalization on MNIST: A From-Scratch Reproduction and Ablation Study

**Author:** Yunhan Zhang
**Date:** <!-- fill in -->

---

## 1. Research Questions

<!--
State these as questions that could turn out either way, not as topics.
For example:

  RQ1 (reproduction) Does a from-scratch 784-30-10 network reproduce the ~95%
      validation accuracy reported by Nielsen, and how large is the
      seed-to-seed variation?
  RQ2 (capacity)     As the hidden layer grows from 10 to 100 units, is there a
      point where additional width stops improving accuracy?
  RQ3 (learning rate) Which qualitatively different training curves appear as
      eta is varied on a logarithmic scale?
  RQ4 (interaction)  Does the benefit of extra capacity depend on the amount of
      training data? Does the larger network do worse when data is scarce?
-->

## 2. Methodology

### 2.1 Implementation
<!-- What was written from scratch: sigmoid activation, quadratic cost,
     backpropagation, mini-batch SGD. State explicitly that only numpy is used
     for the array operations, with no deep learning framework. -->

### 2.2 Data
<!-- MNIST: 50,000 training / 10,000 validation / 10,000 test examples.
     Important sentence: all hyper-parameter comparisons are made on the
     validation set, and the test set is used only once after the final
     configuration is fixed. -->

### 2.3 Experimental Protocol
<!-- Describe what varies and what is held constant in each experiment.
     Each configuration is run with 3 seeds (0, 1, 2); report the mean and the
     spread. A fixed seed controls both the weight initialisation and the
     mini-batch shuffling order. -->

| Experiment | Varied | Held constant |
|---|---|---|
| Baseline | — | hidden=[30], eta=3.0, batch=10, 30 epochs |
| Hidden size | <!-- --> | <!-- --> |
| Learning rate | <!-- --> | <!-- --> |
| Data size x capacity | <!-- --> | <!-- --> |

## 3. Results

### 3.1 Baseline
<!-- Final validation accuracy as mean and spread across seeds, and the epoch
     at which the curve flattens out. Compare with the figure reported in the
     book. -->

### 3.2 Hidden Layer Size
<!-- Figure: ![](../results/figures/hidden_size.png)
     Add a small table giving, for each width, the final accuracy and the
     number of epochs needed to reach 90%. -->

### 3.3 Learning Rate
<!-- Figure, plus a description of the qualitatively different regimes:
     too small to make progress, well matched, and oscillating or diverging. -->

### 3.4 Training Set Size x Network Capacity
<!-- Figure, with attention to whether the curves for different capacities
     cross. -->

## 4. Discussion

<!-- This section carries most of the weight. Do not restate the numbers;
     explain the mechanism behind them:
     - Why do the returns from extra width diminish? (Once representational
       capacity is no longer the bottleneck, the limit shifts to optimisation
       and to the amount of data.)
     - Why does a large eta cause oscillation? (Steps overshoot the minimum
       and the iterates bounce across it.)
     - What does the interaction between data size and capacity suggest?
     - Which result did not match what you expected beforehand? That is
       usually the most valuable paragraph in the report. -->

## 5. Limitations

<!-- Listing these honestly strengthens the report rather than weakening it:
     - Only a quadratic cost with sigmoid units; no cross-entropy, ReLU, or
       regularisation.
     - Only 3 seeds per configuration, so the statistical power is limited.
     - Each experiment varies one factor at a time, so interactions between
       hyper-parameters other than the one studied in 3.4 are not captured.
     - Subsets in the data-efficiency experiment are fixed prefixes rather
       than stratified samples.
     - No early stopping; a fixed budget of 30 epochs may not be equally fair
       to every configuration. -->

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

Raw per-epoch measurements are in `results/metrics.csv`.

## References

1. M. Nielsen, *Neural Networks and Deep Learning*, 2015.
   http://neuralnetworksanddeeplearning.com
2. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning
   applied to document recognition," *Proceedings of the IEEE*, 86(11),
   1998, pp. 2278-2324.
