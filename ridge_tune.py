import pandas as pd
from sklearn.linear_model import Ridge

class RidgeTune:
    def __init__(self, model_configuration: dict[str, any]):
        self.model_configuration = model_configuration

    def train(self, dataset_path: str):
        df = pd.read_csv(dataset_path)
        features = ['rainfall', 'mean_temperature']
        X = df[features]
        Y = df['disease_cases']
        Y = Y.fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
        self.model = Ridge()
        self.model.fit(X, Y)
        print("Train - coefficients: ", list(zip(features, self.model.coef_)))
        return self

    def predict(self, historic_data, future_data):
        df = pd.read_csv(future_data)
        X = df[['rainfall', 'mean_temperature']]
        y_pred = self.model.predict(X)
        df['sample_0'] = y_pred # can write this to file
        print("Predictions: ", y_pred)