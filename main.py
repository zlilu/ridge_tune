import os
import sys
import yaml
from ridge_tune import RidgeTune
from chap_core.datatypes import create_tsdataclass
import joblib
import mlflow, mlflow.sklearn

# minimal template example

def main():
    """
    Train a model using historic data

    Parameters
    ----------
    training_data_filename: str
        The path to the training data file
    model_path: str
        The path to save the trained model
    """
    print("--------------------------start main")
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: train|predict ...")
    cmd, *rest = argv 
    if cmd == "train":
        print("-----------start train")
        training_data_filename: str = rest[0]
        model_path: str = rest[1] # file to store the model? It's "model" not sure what for
        model_config_path: str = rest[2]
        
        model_config = _read_model_config(model_config_path)
        estimator = RidgeTune(model_config)
        predictor = estimator.train(training_data_filename)
        print("----------------pred:", predictor)
        # mlflow.sklearn.log_model(predictor, artifact_path="model")
        joblib.dump(predictor, model_path) # can use MLflow to save info
    elif cmd == "predict":  
        historic_data_path: str = rest[1]
        future_data_path: str = rest[2]
        out_file = rest[3]
        predictor = joblib.load(rest[0])
        predictor.predict(historic_data_path, future_data_path, out_file)
    else:
        raise SystemExit(f"unknown command {cmd}")
    
def _get_dataclass(estimator):
    target: str = "disease_cases"
    data_fields = estimator.covariate_names + [target]
    dc = create_tsdataclass(data_fields)
    return dc

def _read_model_config(model_config_path):
    if model_config_path is not None:
        with open(model_config_path, "r") as file:
            model_config = yaml.safe_load(file)
    else:
        model_config = {}
    return model_config
    # return ModelConfiguration.model_validate(model_config)


if __name__ == "__main__":
    main()