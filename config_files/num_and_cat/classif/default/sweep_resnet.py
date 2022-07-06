import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "resnet_classif_categorical_default",
  "project": "thesis-4",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "log_training": {
      "value": True
    },
    "model__device": {
      "value": "cpu"
    },
    "model_type": {
      "value": "skorch"
    },
    "model_name": {
      "value": "rtdl_resnet"
    },
    "model__use_checkpoints": {
      "value": True
    },
    "model__optimizer": {
      "value": "adamw"
    },
    "model__lr_scheduler": {
      "values": [True]
    },
    "model__batch_size": {
      "values": [512]
    },
    "model__max_epochs": {
      "value": 300
    },
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__normalization": {
      "values": ["batchnorm"]
    },
    "model__module__n_layers": {
      "value": 8,
    },
    "model__module__d": {
      "value": 256,
    },
    "model__module__d_hidden_factor": {
      "value": 2,
    },
    "model__module__hidden_dropout": {
      "value": 0.2,
    },
    "model__module__residual_dropout": {
      "value": 0.2
    },
    "model__lr": {
      "value": 1e-3,
    },
     "model__optimizer__weight_decay": {
      "value": 1e-7,
    },
    "model__module__d_embedding": {
      "value": 128
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 "eye_movements",
                  "KDDCup09_upselling",
                  "covertype",
                  "rl",
                  "road-safety",
                  "compass"]
    },
    "transform__0__method_name": {
      "value": "gaussienize"
    },
    "transform__0__type": {
      "value": "quantile",
    },
    "transform__0__apply_on": {
      "value": "numerical",
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
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")