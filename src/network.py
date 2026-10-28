"""
A feed-forward neural network implemented from scratch with numpy only:
sigmoid activations, a quadratic cost, backpropagation, and mini-batch SGD.

The core algorithm follows Chapter 1 of Michael Nielsen's "Neural Networks and
Deep Learning". This version is ported to Python 3 and extended with the
per-epoch metric logging and seed control needed for controlled experiments.
"""

import random
import time

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes, seed=None):
        """
        sizes: number of neurons per layer, e.g. [784, 30, 10].
        seed:  fixes both the weight initialisation and the mini-batch
               shuffling, so that runs can be compared and reproduced.
        """
        if seed is not None:
            np.random.seed(seed)
        self._shuffler = random.Random(seed)

        self.num_layers = len(sizes)
        self.sizes = sizes
        # randn draws from the standard normal distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the network output for input `a`."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            evaluation_data=None, verbose=True):
        """
        Train the network using mini-batch stochastic gradient descent.

        eta:             learning rate
        evaluation_data: evaluated after every epoch. During hyper-parameter
                         tuning this should be the validation set, never the
                         test set.

        Returns a history list with one record per epoch, which the experiment
        scripts write to CSV and use to plot training curves.
        """
        # Work on a copy so that shuffling does not change the ordering seen by
        # other runs that share the same data object.
        training_data = list(training_data)
        n = len(training_data)
        n_eval = len(evaluation_data) if evaluation_data else 0

        history = []
        t0 = time.perf_counter()

        for j in range(epochs):
            self._shuffler.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            record = {"epoch": j, "seconds": round(time.perf_counter() - t0, 2)}

            if evaluation_data:
                correct = self.evaluate(evaluation_data)
                record["eval_correct"] = correct
                record["eval_total"] = n_eval
                record["eval_accuracy"] = correct / n_eval
                if verbose:
                    print(f"Epoch {j}: {correct} / {n_eval}  ({correct/n_eval:.4f})")
            elif verbose:
                print(f"Epoch {j} complete")

            history.append(record)

        return history

    def update_mini_batch(self, mini_batch, eta):
        """
        Apply one gradient descent step using the examples in `mini_batch`.
        `nabla` denotes a gradient.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Gradient contributed by this single training example.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Accumulate the gradients over the whole mini-batch.
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update rule:  w <- w - (eta / m) * grad_w,  where m = mini-batch size
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Backpropagation: compute the gradient of the cost for one example.
        Returns (nabla_b, nabla_w), layer by layer.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass: store every activation and every weighted input z.
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass: start from the output layer error, then propagate it
        # back one layer at a time.
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Count how many examples the network classifies correctly."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Partial derivatives dC_x/da for the output activations."""
        return (output_activations - y)
