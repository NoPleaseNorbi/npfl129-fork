#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        target = train.target
        data = train.data
        data = np.c_[data, np.ones(data.shape[0])]
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=0.6, random_state=args.seed)
        weights = np.random.RandomState(args.seed).uniform(size=train_data.shape[1], low=-0.5, high=0.5)
        epochs = 100000
        batch_size = 30
        learning_rate = 0.9
        for epoch in range(epochs):
            permutation = np.random.RandomState(args.seed).permutation(train_data.shape[0])
        
            for i in range(0, train_data.shape[0], batch_size):
                permutate_train_data = train_data[permutation[i:i+batch_size]]
                permutate_train_target = train_target[permutation[i:i+batch_size]]
    
                sigmoid = 1 / (1 + np.exp(-(np.matmul(permutate_train_data, weights))))
                errors = sigmoid - permutate_train_target
                gradient = np.matmul(errors, permutate_train_data)
                average = gradient / batch_size
                weights = weights - learning_rate * average
                
            pred_prod_train_data = 1 / (1 + np.exp(-(np.matmul(train_data, weights))))
            pred_prod_test_data = 1 / (1 + np.exp(-(np.matmul(test_data, weights))))
            pred_prod_train_data_reduced = (np.array(pred_prod_train_data) > 0.5).astype(int)
            pred_prod_test_data_reduced = (np.array(pred_prod_test_data) > 0.5).astype(int)
    
            train_accuracy = sklearn.metrics.accuracy_score(train_target, pred_prod_train_data_reduced)
            test_accuracy = sklearn.metrics.accuracy_score(test_target, pred_prod_test_data_reduced)
            train_loss = sklearn.metrics.log_loss(train_target, pred_prod_train_data)
            test_loss = sklearn.metrics.log_loss(test_target, pred_prod_test_data)
            print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
                        epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))
        
        
        model = {
            "weights": weights,
        }

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = (1 / (1 + np.exp(-(np.matmul(np.c_[test.data, np.ones(test.data.shape[0])], model["weights"])))) > 0.5).astype(int)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
