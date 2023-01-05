import pandas as pd
import os
import json
import glob
import argparse


openml_ids = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131, 44089, 44090, 44091]

def save_formatted_df(restore_path, df):
    df["data__keyword"] = df["dataset_name"]
    df["mean_test_score"] = df["test"]
    df["mean_train_score"] = df["train"]
    df["mean_val_score"] = df["val"]
    df["min_test_score"] = df["test"]
    df["min_train_score"] = df["train"]
    df["min_val_score"] = df["val"]
    df["max_test_score"] = df["test"]
    df["max_train_score"] = df["train"]
    df["max_val_score"] = df["val"]
    df["max_train_samples"] = 10_000
    df["data__categorical"] = 0.0
    df["data__method_name"] = "real_data"
    df["data__regression"] = 0.0
    df["std_test_score"] = 0
    df["std_train_score"] = 0
    df["std_val_score"] = 0
    df["mean_time"] = df["duration"]
    df["std_time"] = 0
    df["train_prop"] = 0.7
    df["val_test_prop"] = 0.3
    df["benchmark"] = "numerical_classif_medium"
    df["num_epochs"] = 105

    null_columns = ["mean_r2_test", "mean_r2_val", "...7", "_runtime", "_step", "_timestamp", "_wandb", "data_generation_time", "max_test_samples", "max_val_samples", "mean_r2_train", "model_type", "n_features", "transformed_target",
                "n_iter", "n_test", "n_train", "one_hot_encoder", "processor", "regression", "std_r2_test", "std_r2_train", "std_r2_val", "step", "model__criterion", "model__learning_rate", "model__loss", "model__max_depth",
                "model__max_leaf_nodes", "model__min_impurity_decrease", "model__min_samples_leaf", "model__min_samples_split", "model__n_estimators", "model__n_iter_no_change", "model__subsample", "model__validation_fraction",
                "early_stopping_rounds", "model__colsample_bylevel", "model__colsample_bytree", "model__gamma", "model__min_child_weight", "model__reg_alpha", "model__reg_lambda", "model__use_label_encoder", "model__early_stopping",
                "model__max_iter", "model__bootstrap", "model__max_features", "log_training", "model__batch_size", "model__device", "model__lr", "model__lr_scheduler", "model__max_epochs", "model__module__d_embedding",
                "model__module__d_layers", "model__module__dropout", "model__module__n_layers", "model__optimizer", "model__use_checkpoints", "transform__0__apply_on", "transform__0__method_name", "transform__0__type",
                "model__module__activation", "model__module__d", "model__module__d_hidden_factor", "model__module__hidden_dropout", "model__module__normalization", "model__module__residual_dropout", "model__optimizer__weight_decay",
                "d_token", "model__module__attention_dropout", "model__module__d_ffn_factor", "model__module__d_token", "model__module__ffn_dropout", "model__module__initialization", "model__module__kv_compression",
                "model__module__kv_compression_sharing", "model__module__n_heads", "model__module__prenormalization", "model__module__token_bias", "model__args__batch_size", "model__args__data_parallel", "model__args__early_stopping_rounds",
                "model__args__epochs", "model__args__lr", "model__args__model_name", "model__args__num_classes", "model__args__objective", "model__args__use_gpu", "model__args__val_batch_size", "model__params__depth", "model__params__dim",
                "model__params__dropout", "model__params__heads", "...129", "Name", "Agent", "State", "Notes", "User", "Tags", "Created", "Runtime", "Sweep", "data__n_features", "data__n_samples", "data__num_samples", "model__clf_loss",
                "model__module_d_token", "model__n_features_per_subset", "model__rotation_algo", "model__wandb_run", "model__weight_decay", "model_module__initialization", "model_module__prenormalization", "target__method_name",
                "target__n_periods", "target__noise", "target__period", "target__period_size", "train_set_prop", "transform__0__max_rel_decrease", "transform__0__multiplier", "transform__1__method_name", "transform__1__multiplier",
                "transform__1__type", "transform__2__method_name", "transform__2__type", "...165", "transform__0__n_iter", "transform__1__num_features", "transform__2__cov_mult", "transform__2__covariance_estimation", "transform__2__deactivated",
                "...171", "...172", "model__alpha", "model__class_weight", "transform__0__model_to_use", "transform__0__num_features_to_remove", "transform__0__keep_removed_features", "transform__1__max_rel_decrease", "transform__1__n_iter",
                "...180", "...181", "...1", "train_accuracy_vector", "valid_accuracy_vector","valid_loss_vector","test_score","train_score", "test_scores", "times", "train_scores", "val_scores"
                ]

    df = df.assign(**dict(zip(null_columns, [pd.NA]*len(null_columns))))
    df["Unnamed: 0"] = pd.NA
    df.to_csv(restore_path.replace(".csv", "_formatted.csv"), index=False)
    their_df = pd.read_csv("analyses/results/random_search_benchmark_numerical.csv", index_col=None)
    df = df[their_df.columns]
    df.to_csv(restore_path.replace(".csv", "_formatted.csv"), index=False)


def get_traj_random(search_path, filter_benchmark_ids):
    trajectories_glob = list()
    for file in glob.glob(search_path, recursive=True):
        runhistory = json.load(open(os.path.join(file), 'r'))
        dataset_info = json.load(open(os.path.join(os.path.dirname(file), '../result.json'), 'r'))
        dataset_id = dataset_info.get('dataset_id', None)
        if filter_benchmark_ids and dataset_id not in openml_ids:
            continue
        runs = []
        for key, value in runhistory.items():
            if key == 1:
                hp = "default"
            else:
                hp = "random"
            runs.append({'config_id': key, 'duration': value.get("duration", 0), **value["cost"], 'model_name': 'cocktails_random', 'dataset_name': dataset_info.get('dataset_name', None), 'dataset_id': dataset_id, "hp": hp})
        run_trajectory = pd.DataFrame(runs)
        trajectories_glob.append(run_trajectory)
    return trajectories_glob

def get_traj_smac(search_path, filter_benchmark_ids):
    trajectories_glob = list()
    print(f"Search path is: {search_path}")
    for file in glob.glob(search_path, recursive=True):
        print(f"In the loop found: {file}")
        runhistory = json.load(open(file, 'r'))
        runhistory_data = runhistory["data"]
        dataset_info = json.load(open(os.path.join(os.path.dirname(file), '../../../result.json'), 'r')) # runhistory now is inside tmp/smac3-output/run_1/
        dataset_id = dataset_info.get('dataset_id', None)
        if filter_benchmark_ids and dataset_id not in openml_ids:
            continue
        runs = []
        for run_key, run_info in runhistory_data:
            _, _, status_type, _, _, loss_dict = tuple(run_info)
            status_type = status_type['__enum__']
            if "success" not in status_type.lower():
                continue
            hp = loss_dict.get("configuration_origin", "smac").lower()
            runs.append({'config_id': run_key[0], 'duration': loss_dict.get("duration", 0), 'val': 1 - loss_dict.get('accuracy', 1), 'train': 1 - loss_dict.get('train_loss', {}).get('accuracy', 1), 'test': 1 - loss_dict.get('test_loss', 1), 'model_name': 'cocktails_smac', 'dataset_name': dataset_info.get('dataset_name', None), 'dataset_id': dataset_id, "hp": hp})
        run_trajectory = pd.DataFrame(runs)
        trajectories_glob.append(run_trajectory)
    return trajectories_glob

def get_trajectories(search_path: str, restore_path: str, filter_benchmark_ids: bool = False):
    if os.path.exists(restore_path):
        return pd.read_csv(restore_path)
    else:
        func = get_traj_smac if 'smac' in search_path else get_traj_random
        trajectories_glob = func(search_path, filter_benchmark_ids)
        df = pd.concat(trajectories_glob)
        df.to_csv(restore_path, index=False)
        return df

parser = argparse.ArgumentParser()

parser.add_argument(
    "--search_path",
    type=str,
    help="Path where run history is stored",
    default="autopytorch_cocktails_random_runs_10k_no_preprocess"
)
parser.add_argument(
    "--restore_path",
    type=str,
    help="output path to store the data",
    default="medium_tasks_cocktails_random_no_preprocess.csv"
)
parser.add_argument(
    "--filter_benchmark_ids",
    action="store_true",
    help="Path where run history is stored"
)
args = parser.parse_args()
runhistory_file_name = "runhistory.json" if "smac" in args.search_path else "run_history.json"
search_path = os.path.join("/work/dlclarge2/rkohli-results_tab-bench/", args.search_path, "**", runhistory_file_name)
restore_path = args.restore_path.replace(".csv", "_filtered.csv") if args.filter_benchmark_ids else args.restore_path
restore_path = os.path.join("/home/rkohli/tabular-benchmark/", restore_path)

if __name__ == "__main__":
    df = get_trajectories(search_path, restore_path, args.filter_benchmark_ids)

    save_formatted_df(restore_path, df)

