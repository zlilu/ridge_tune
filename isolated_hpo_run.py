from predict import predict
from train import train
import yaml
import pandas as pd
from sklearn.metrics import root_mean_squared_error


config = yaml.safe_load(open("config_hpo.yaml"))
best_score = -float("inf")
best_model = None
model_file = "output/model.bin"
predictions_file = "output/predictions.csv"
for alpha in config["alpha"]["values"]:
    model_config = config.copy()
    model_config["alpha"] = alpha
    model_config = {"user_option_values": model_config}
    # model_file = f"output/model_{alpha}.bin"
    validation_file = "input/validation.csv"
    train("input/train.csv", model_file, model_config)
    # rmse_score = predict(model_file, "input/train.csv", validation_file, predictions_file)
    
    predict(model_file, "input/train.csv", validation_file, predictions_file)
    df_val = pd.read_csv(validation_file)
    y_val = df_val['disease_cases']
    df_pred = pd.read_csv(predictions_file)
    y_pred = df_pred['sample_0']
    rmse_score = -root_mean_squared_error(y_val, y_pred)
    
    print("rmse:", rmse_score)
    if rmse_score > best_score:
        best_score = rmse_score
        best_model = model_file
print("best score", best_score)

#final prediction..
# rmse_score = predict(best_model, "input/train.csv", "input/test.csv", "output/predictions.csv")
test_file = "input/test.csv"
predict(best_model, "input/train.csv", test_file, predictions_file)
df_test = pd.read_csv(test_file)
y_test = df_test['disease_cases']
df_pred = pd.read_csv(predictions_file)
y_pred = df_pred['sample_0']
rmse_score = -root_mean_squared_error(y_test, y_pred)

print("final test score:", rmse_score)