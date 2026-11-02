"""Load the MNIST dataset and reshape it for the network."""

import gzip
import pickle
from pathlib import Path

import numpy as np

# Resolved relative to the repository root so that the experiment scripts work
# no matter which directory they are run from.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mnist.pkl.gz"


def load_data():
    """Return the raw (training, validation, test) tuples from the pickle."""
    with gzip.open(DATA_PATH, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Return the three sets in the format the network expects.

    Training labels are one-hot vectors of shape (10, 1); validation and test
    labels stay as plain integers, since they are only used for counting
    correct predictions.
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]    # Reshape each 28x28 image into a 784x1 column vector
    training_results = [vectorized_result(y) for y in tr_d[1]]      # Convert labels to one-hot vectors
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Turn a digit label 0..9 into a one-hot column vector."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
