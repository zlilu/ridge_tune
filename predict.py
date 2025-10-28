import argparse
import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def predict(model_fn: str, historic_data: str, future_data: str, out_file: str): # model_config is given in MLproject not sure why
    df = pd.read_csv(future_data)
    X = df[['rainfall', 'mean_temperature']]
    # y_val = df['disease_cases'] # this must happen outside otherwise chap-core errors, prop bc dataset sent to predict is without gold label
    model = joblib.load(model_fn)
    y_pred = model.predict(X)
    df['sample_0'] = y_pred # not sure why return future_data and sample_0
    df.to_csv(out_file, index=False)
    # print("Predictions: ", y_pred)
    # return -root_mean_squared_error(y_val, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')
    parser.add_argument('model_config', type=str, help='Path to model configuration yaml.') # not used not sure why needed

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)