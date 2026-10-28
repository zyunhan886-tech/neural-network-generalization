"""
手写实现的全连接神经网络 + SGD + 反向传播。
基于 Michael Nielsen《Neural Networks and Deep Learning》第 1 章，
移植到 Python 3 并加入实验所需的指标记录与随机种子控制。
"""

import random
import time

import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes, seed=None):
        # seed: 固定权重初始化和 mini-batch 洗牌，保证对照实验可复现
        if seed is not None:
            np.random.seed(seed)
        self._shuffler = random.Random(seed)

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # np.random.randn: return a sample (or samples) from the "standard normal" distribution.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):     # zip is a function from python that it repairs two things.
            a = sigmoid(np.dot(w, a)+b)
        return a


    """SGD: Stochastic Gradient Descent"""
    def SGD(self, training_data, epoches, mini_batch_size, eta,
            evaluation_data=None, verbose=True):
        # epoches: number of epochs to train for
        # mini_batch_size: size of mini-batches to use when sampling
        # eta: learning rate
        # evaluation_data: 每个 epoch 结束后用来算准确率的数据集（调参阶段应传 validation_data）
        #
        # 返回 history: 每个 epoch 一条记录，供实验脚本写入 CSV / 画训练曲线

        training_data = list(training_data)   # 复制一份，避免原地洗牌影响同一数据上的其它 run
        n = len(training_data)
        n_eval = len(evaluation_data) if evaluation_data else 0

        history = []
        t0 = time.perf_counter()

        for j in range(epoches):
            self._shuffler.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
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


    """backpropagation algorithm"""
    """a fast way of computing the gradient of the cost function"""
    def update_mini_batch(self, mini_batch, eta):
        # .shape means the shape of the array, # rows, # columns
        # nabla means the gradient
        nabla_b = [np.zeros(b.shape) for b in self.biases]    # 全0数组
        nabla_w = [np.zeros(w.shape) for w in self.weights]    # 全0数组

        for x, y in mini_batch:
            # delta means 这一次新算出来的
            """Most of the work is done by the line"""
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # calculate the gradient

            # nable_b and nable_w are the sum of all the gradients in the mini-batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 旧+新
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw      # formula: w new = w old - (eta / m) * ∇w
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                    for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return (output_activations-y)
