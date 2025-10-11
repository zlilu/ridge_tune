import argparse

import joblib
import pandas as pd
from sklearn.linear_model import Ridge


def train(self, dataset_path: str, model_configuration: dict[str, any]): # filname yaml
        print("----------------------Training with alpha:", self.alpha)
        df = pd.read_csv(dataset_path)
        features = ['rainfall', 'mean_temperature']
        X = df[features]
        Y = df['disease_cases']
        Y = Y.fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, Y)
        print("Train - coefficients: ", list(zip(features, self.model.coef_)))
        print("returning myself:", type(self))
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)

