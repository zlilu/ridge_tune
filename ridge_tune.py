import pandas as pd
from sklearn.linear_model import Ridge

class RidgeTune: # dont need a class only two methods train and predict
    def __init__(self, model_configuration: dict[str, any]):
        self.model_configuration = model_configuration
        self.alpha = float(model_configuration["user_option_values"]["alpha"]) # TODO: skip validation for "user_options" in yaml/dic

    def train(self, dataset_path: str): # filname yaml
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

    def predict(self, historic_data, future_data, out_file):
        df = pd.read_csv(future_data)
        X = df[['rainfall', 'mean_temperature']]
        y_pred = self.model.predict(X)
        df['sample_0'] = y_pred # can write this to file
        print("Predictions: ", y_pred)
        df.to_csv(out_file, index=False)
        print("-----wrote to:", out_file)