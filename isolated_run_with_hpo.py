from predict import predict
from train import train

config = _read_model_config(model_config_path)
for alpha in config.alpha.values:
    specific_config = copy(config)
    specific_config.alpha.values = alpha
    model_file = f"output/model_{alpha}.bin"
    train("input/trainData.csv", model_file, specific_config)
    predict("output/model.bin", "input/trainData.csv", "input/futureClimateData.csv", "output/predictions.csv")
    score = compute_score("output/predictions.csv","input/validation_truth.csv")
    if score > best:
        best = score
        best_model = model_file

#final prediction..
predict(best_model, "input/trainData.csv", "input/futureClimateData.csv", "output/predictions.csv")
score = compute_score("output/predictions.csv","input/test_truth.csv")
