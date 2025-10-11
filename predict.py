import argparse

import joblib
import pandas as pd

def predict(self, historic_data, future_data, out_file):
        df = pd.read_csv(future_data)
        X = df[['rainfall', 'mean_temperature']]
        y_pred = self.model.predict(X)
        df['sample_0'] = y_pred # can write this to file
        print("Predictions: ", y_pred)
        df.to_csv(out_file, index=False)
        print("-----wrote to:", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)