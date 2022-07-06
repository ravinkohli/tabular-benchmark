import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "rf_categorical_classif_large",
  "project": "thesis-4",
  "method" : "random",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "rf_c"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": {
      "distribution": "q_log_uniform_values",
      "min": 9.5,
      "max": 3000.5,
      "q": 1
    },
    "model__criterion": {
      "values": ["gini", "entropy"],
    },
    "model__max_features": { # like Hyperopt ?
      "values": ["sqrt", "sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    "model__max_depth": { # Not exactly like Hyperopt
      "values": [None, 2, 3, 4],
      "probabilities": [0.7, 0.1, 0.1, 0.1]
    },
    "model__min_samples_split": {
      "values": [2, 3],
      "probabilities": [0.95, 0.05]
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_log_uniform_values",
      "min": 1.5,
      "max": 50.5,
      "q": 1
    },
    "model__bootstrap": {
      "values": [True, False]
    },
    "model__min_impurity_decrease": {
        "values": [0.0, 0.01, 0.02, 0.05],
        "probabilities": [0.85, 0.05, 0.05, 0.05]
    },
      "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 #"eye_movements",
                 # "KDDCup09_upselling",
                  "covertype",
                 # "rl",
                  "road-safety"]
                  #"compass"]
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
          "value": False
    },
    "data__regression": {
          "value": False
    },
    "data__categorical": {
            "value": True
    },
    "one_hot_encoder": {
      "value": True
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")