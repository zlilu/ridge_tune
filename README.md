# Minimal model template example 
This document demonstrates a minimal example of a model template using Scikit-learn's Ridge Regression with tunable hyperparameter alpha. The example is written in Python, uses few variables without any lag. It simply learns a linear regression from rain and temperature to disease cases, without considering any previous disease or climate data. It also assumes and works only with a single region. The model is not meant to accurately capture any interesting relations - the purpose is just to show how CHAP integration works in a simplest possible setting. Allows for running with or without hyperparameter optimization.

## Running the model without CHAP integration
Before getting a new model to work as part of CHAP, it can be useful to develop and debug it while running it directly a small dataset from file.

The example can be run in isolation (e.g. from the command line) using the file isolated_run.py:

```bash
python3 isolated_run.py
```

### Trainning data 

## Running the minimalist model as part of CHAP
To run the minimalist model in CHAP, we first define the model interface in an MLFlow-based yaml specification (in the file "MLproject", which defines :
```yaml
name: min_template_py_ex

entry_points:
  train:
    command: python train.py {train_data} {model} {model_config}
    parameters:
      train_data: str
      model: str
      model_config: str
  predict:
    command: python predict.py {model} {historic_data} {future_data} {out_file} {model_config}
    parameters:
      model: str
      historic_data: str
      future_data: str
      out_file: str
      model_config: str
python_env: pyenv.yaml
```

After you have installed chap-core (see here for installation instructions: https://github.com/dhis2-chap/chap-core), you can run this minimalist model through CHAP as follows (remember to replace '/path/to/your/model/directory' with your local path):
```bash
chap evaluate-hpo --model-name /path/to/your/model/directory --dataset-name hydromet_5_filtered --model-configuration-yaml /path/to/config/yaml/file --report-filename report.pdf --ignore-environment  --debug
```
