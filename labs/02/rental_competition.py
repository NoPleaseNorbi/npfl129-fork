#!/usr/bin/env python3
#TEAM MEMBERS:
# a652f90b-2220-11ec-986f-f39926f24a9c
# 4ae50bd7-2516-11ec-986f-f39926f24a9c
# 5212d2e4-2516-11ec-986f-f39926f24a9c
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
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
        target = train.target
        data = train.data

        # TODO: Train a model on the given dataset and store it in `model`.
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            data, target, test_size=0.2, random_state=args.seed
        )

        categorical_columns = np.arange(0, 8)
        numerical_columns = np.arange(8, 12)
        transformer = sklearn.compose.ColumnTransformer(transformers=[
                ('t1',sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_columns),
                ('t2', sklearn.preprocessing.StandardScaler(), numerical_columns)
                ])
        transformed_train_data = transformer.fit_transform(train_data)
        transformed_test_data = transformer.transform(test_data)
        preprocessor = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
        preprocessed_train_data = preprocessor.fit_transform(transformed_train_data)
        preprocessed_test_data = preprocessor.transform(transformed_test_data)
        weights = np.zeros(preprocessed_train_data.shape[1])
        epochs = 200
        batch_size = 30
        learning_rate = 0.001
        l2 = 0.0001
        for epoch in range(epochs):
            permutation = np.random.permutation(preprocessed_train_data.shape[0])

            for i in range(0, preprocessed_train_data.shape[0], batch_size):
                weights_with_bias_set_to_zero = weights.copy()
                permutate_train_data = preprocessed_train_data[permutation[i:i+batch_size]]
                permutate_train_target = train_target[permutation[i:i+batch_size]]

                weight_helper = np.matmul(permutate_train_data, weights)
                errors = weight_helper - permutate_train_target
                gradient = np.matmul(errors, permutate_train_data)
                average = gradient / len(permutate_train_target)
                average += l2 * weights_with_bias_set_to_zero
                weights = weights - learning_rate * average

            print(np.sqrt(np.mean((np.matmul(preprocessed_test_data, weights) - test_target)**2)))



        model = {
            "weights": weights,
            "transformer": transformer,
            "polynomial": preprocessor
        }
        rmse = np.sqrt(np.mean((np.matmul(preprocessed_test_data, weights) - test_target)**2))
        print(rmse)
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        test_data = model["transformer"].transform(test.data)
        test_data = model["polynomial"].transform(test_data)
        predictions = np.matmul(test_data, model["weights"])
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
