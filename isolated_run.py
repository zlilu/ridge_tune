from predict import predict
from train import train

train("input/train.csv", "output/model.bin")
predict("output/model.bin", "input/train.csv", "input/futureClimateData.csv", "output/predictions.csv")
