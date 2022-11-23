from __future__ import annotations
import os

os.environ["PROJECT_DIR"] = "test"
import openml

import sys
from typing import Any, Dict

sys.path.append(".")
print(sys.path)
import torch
import pandas as pd
from train import *
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# import train test split
from sklearn.model_selection import train_test_split
import argparse
import traceback
from preprocessing.preprocessing import preprocessing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
openml.datasets.functions._get_dataset_parquet = lambda x: None #to bypass current OpenML issues #TODO remove

from reproduce_utils import set_seed, get_executer, arguments, Dataset


def run_on_dataset(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | pd.Series,
    categorical_indicator: list[bool],
    args: argparse.Namespace,
    dataset_id: int,
    dataset_name: str,
    resnet_config: dict
) -> dict[str, Any]:
    transformation = None
    # if "Transformation" in row.keys():
    #     if not pd.isnull(row["Transformation"]):
    #         transformation = row["Transformation"]

    X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
    num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
        preprocessing(X, y, categorical_indicator, categorical=args.categorical,
                        regression=args.regression, transformation=transformation)
    train_prop = 0.7
    train_prop = min(15000 / X.shape[0], train_prop)
    numeric_transformer = StandardScaler()
    numeric_transformer_sparse = MaxAbsScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
        ]
    )
    preprocessor_sparse = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_sparse,
                [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
        ]
    )

    score_resnet_list = []
    score_linear_list = []
    score_hgbt_list = []
    score_tree_list = []
    score_mlp_list = []
    score_rf_list = []
    if int((1 - train_prop) * X.shape[0]) > 10000:
        n_iters = 1
    elif int((1 - train_prop) * X.shape[0]) > 5000:
        n_iters = 3
    else:
        n_iters = 5
    print(X.shape)
    if X.shape[0] > 3000 and X.shape[1] > 3:
        score_resnet, score_linear, score_hgbt, score_tree, score_mlp, score_rf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        for iter in range(n_iters):
            set_seed(seed=iter)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop,
                                                                random_state=np.random.RandomState(iter))
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(
                y_train), np.array(y_test)
            if resnet_config["regression"] == True:
                y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
                y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
            if X_test.shape[0] > 30000:  # for speed
                indices = np.random.choice(X_test.shape[0], 30000, replace=False)
                try:
                    X_test = X_test.iloc[indices]
                except:
                    X_test = X_test[indices]
                y_test = y_test[indices]
            try:
                X_train_one_hot = preprocessor.fit_transform(X_train)
                X_test_one_hot = preprocessor.transform(X_test)
                X_train_no_one_hot = np.zeros_like(X_train)
                X_test_no_one_hot = np.zeros_like(X_test)
                # not column transformer to preserve order
                for i in range(X_train.shape[1]):
                    if categorical_indicator[i]:
                        X_train_no_one_hot[:, i] = X_train[:, i]
                        X_test_no_one_hot[:, i] = X_test[:, i]
                    else:
                        X_train_no_one_hot[:, i] = numeric_transformer.fit_transform(
                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                        X_test_no_one_hot[:, i] = numeric_transformer.transform(
                            X_test[:, i].reshape(-1, 1)).reshape(-1)

            except:
                print("trying MaxAbsScaler")
                X_train_one_hot = preprocessor_sparse.fit_transform(X_train)
                X_test_one_hot = preprocessor_sparse.transform(X_test)
                X_train_no_one_hot = np.zeros_like(X_train)
                X_test_no_one_hot = np.zeros_like(X_test)
                for i in range(X_train.shape[1]):
                    if categorical_indicator[i]:
                        X_train_no_one_hot[:, i] = X_train[:, i]
                        X_test_no_one_hot[:, i] = X_test[:, i]
                    else:
                        X_train_no_one_hot[:, i] = numeric_transformer_sparse.fit_transform(
                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                        X_test_no_one_hot[:, i] = numeric_transformer_sparse.transform(
                            X_test[:, i].reshape(-1, 1)).reshape(-1)

            y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
            X_train_one_hot, X_test_one_hot = np.array(X_train_one_hot), np.array(X_test_one_hot)
            X_train_no_one_hot, X_test_no_one_hot = np.array(X_train_no_one_hot), np.array(
                X_test_no_one_hot)
            X_train_one_hot, X_test_one_hot = X_train_one_hot.astype(np.float32), X_test_one_hot.astype(
                np.float32)
            X_train_no_one_hot, X_test_no_one_hot = X_train_no_one_hot.astype(
                np.float32), X_test_no_one_hot.astype(np.float32)

            if args.regression:
                if not "linear" in args.remove_model:
                    linear_model = TransformedTargetRegressor(regressor=LinearRegression(),
                                                            transformer=QuantileTransformer(
                                                                output_distribution="normal"))
                if not "hgbt" in args.remove_model:
                    hgbt = HistGradientBoostingRegressor(categorical_features=categorical_indicator)
                if not "tree" in args.remove_model:
                    tree = DecisionTreeRegressor()
                if not "mlp" in args.remove_model:
                    mlp = MLPRegressor()
                if not "rf" in args.remove_model:
                    rf = RandomForestRegressor()
            else:
                if not "linear" in args.remove_model:
                    linear_model = LogisticRegression()
                if not "hgbt" in args.remove_model:
                    hgbt = HistGradientBoostingClassifier(categorical_features=categorical_indicator)
                if not "tree" in args.remove_model:
                    tree = DecisionTreeClassifier()
                if not "mlp" in args.remove_model:
                    mlp = MLPClassifier()
                if not "rf" in args.remove_model:
                    rf = RandomForestClassifier()

            if not "linear" in args.remove_model:
                linear_model.fit(X_train_one_hot, y_train)
            if not "hgbt" in args.remove_model:
                hgbt.fit(X_train_no_one_hot, y_train)
            if not "tree" in args.remove_model:
                tree.fit(X_train_one_hot, y_train)
            if not "mlp" in args.remove_model:
                mlp.fit(X_train_one_hot, y_train)
            if not "rf" in args.remove_model:
                rf.fit(X_train_one_hot, y_train)
            if args.regression:
                if not "linear" in args.remove_model:
                    score_linear = -mean_squared_error(y_test, linear_model.predict(X_test_one_hot),
                                                        squared=False)
                print("Linear model score: ", score_linear)
                if not "hgbt" in args.remove_model:
                    score_hgbt = -mean_squared_error(y_test, hgbt.predict(
                        X_test_no_one_hot), squared=False)
                    print("HGBT score: ", score_hgbt)
                if not "tree" in args.remove_model:
                    score_tree = -mean_squared_error(y_test, tree.predict(
                        X_test_one_hot), squared=False)
                    print("Tree score: ", score_tree)
                if not "mlp" in args.remove_model:
                    score_mlp = -mean_squared_error(y_test, mlp.predict(
                        X_test_one_hot), squared=False)
                    print("MLP score: ", score_mlp)
                if not "rf" in args.remove_model:
                    score_rf = -mean_squared_error(y_test, rf.predict(
                        X_test_one_hot), squared=False)
                    print("rf score: ", score_rf)
            else:
                if not "linear" in args.remove_model:
                    score_linear = linear_model.score(X_test_one_hot, y_test)  # accuracy
                    print("Linear model score: ", score_linear)
                if not "hgbt" in args.remove_model:
                    score_hgbt = hgbt.fit(X_train_no_one_hot, y_train).score(X_test_no_one_hot, y_test)
                    print("HGBT score: ", score_hgbt)
                if not "tree" in args.remove_model:
                    score_tree = tree.fit(X_train_one_hot, y_train).score(X_test_one_hot, y_test)
                    print("Tree score: ", score_tree)
                if not "mlp" in args.remove_model:
                    score_mlp = mlp.fit(X_train_one_hot, y_train).score(X_test_one_hot, y_test)
                    print("MLP score: ", score_mlp)
                if not "rf" in args.remove_model:
                    score_rf = rf.fit(X_train_one_hot, y_train).score(X_test_one_hot, y_test)
                    print("RF score: ", score_rf)
                

            if not "resnet" in args.remove_model:
                if resnet_config["regression"] == True:
                    y_train = y_train.reshape(-1, 1)
                    y_test = y_test.reshape(-1, 1)
                    y_train, y_test = y_train.astype(np.float32), y_test.astype(
                        np.float32)
                else:
                    y_train = y_train.reshape(-1)
                    y_test = y_test.reshape(-1)
                    print("Number of classes: ", len(np.unique(y_train)))
                    print("Number of classes max: ", np.max(y_train))
                # Give the true number of categories to the model
                categories = []
                for i in range(len(categorical_indicator)):
                    if categorical_indicator[i]:
                        categories.append(int(np.max(X.iloc[:, i]) + 1))
                resnet_config["model__categories"] = categories
                model, model_id = train_model(iter, X_train_no_one_hot, y_train, categorical_indicator if len(categorical_indicator) > 0 else None,
                                                resnet_config)
                train_score, val_score, score_resnet = evaluate_model(model, X_train_no_one_hot, y_train, None,
                                                                        None, X_test_no_one_hot,
                                                                        y_test, resnet_config, model_id,
                                                                        return_r2=False)
                if args.regression:
                    score_resnet = -score_resnet  # we want high = good so we take -RMSE

            score_resnet_list.append(score_resnet)
            score_linear_list.append(score_linear)
            score_hgbt_list.append(score_hgbt)
            score_tree_list.append(score_tree)
            score_mlp_list.append(score_mlp)
            score_rf_list.append(score_rf)
            print("resnet score: ", score_resnet)
            print("linear score: ", score_linear)
            print("hgbt score: ", score_hgbt)
            print("tree score: ", score_tree)
            print("mlp score: ", score_mlp)
            print("rf score: ", score_rf)
        print("Linear score: {}".format(score_linear_list))
        print("Resnet score: {}".format(score_resnet_list))
        print("HGBT score: {}".format(score_hgbt_list))
        print("Tree score: {}".format(score_tree_list))
        print("MLP score: ", score_mlp_list)
        print("RF score: ", score_rf_list)
        if args.regression:
            score_linear = np.nanmedian(score_linear_list)
            score_resnet = np.nanmedian(score_resnet_list)
            score_hgbt = np.nanmedian(score_hgbt_list)
            score_tree = np.nanmedian(score_tree_list)
            score_rf = np.nanmedian(score_rf_list)
            score_mlp = np.nanmedian(score_mlp_list)
        else:
            score_linear = np.mean(score_linear_list)
            score_resnet = np.mean(score_resnet_list)
            score_hgbt = np.mean(score_hgbt_list)
            score_tree = np.mean(score_tree_list)
            score_rf = np.mean(score_rf_list)
            score_mlp = np.mean(score_mlp_list)


        res_dic = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "original_n_samples": original_n_samples,
            "original_n_features": original_n_features,
            "num_categorical_columns": num_categorical_columns,
            "num_pseudo_categorical_columns": n_pseudo_categorical,
            "num_columns_missing": num_columns_missing,
            "num_rows_missing": num_rows_missing,
            # "too_easy": too_easy,
            "score_resnet": score_resnet,
            "score_linear": score_linear,
            "score_hgbt": score_hgbt,
            "score_tree": score_tree,
            "score_mlp": score_mlp,
            "score_rf": score_rf,
            "heterogeneous": pd.NA,
            "n_samples": X.shape[0],
            "too_small": False}

    else:
        print("dataset too small after preprocessing")
        res_dic = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "original_n_samples": original_n_samples,
            "original_n_features": original_n_features,
            "num_categorical_columns": num_categorical_columns,
            "num_pseudo_categorical_columns": n_pseudo_categorical,
            "num_columns_missing": num_columns_missing,
            "num_rows_missing": num_rows_missing,
            # "too_easy": pd.NA,
            "score_resnet": pd.NA,
            "score_linear": pd.NA,
            "score_hgbt": pd.NA,
            "score_tree": pd.NA,
            "score_mlp": pd.NA,
            "score_rf": pd.NA,
            "heterogeneous": pd.NA,
            "n_samples": X.shape[0],
            "too_small": True}

    return res_dic


def do_evaluations_parallel(
    datasets: list[Dataset],
    args: argparse.Namespace,
    resnet_config: dict,
    device: str = "cpu",
    only_fetch_data: bool = False
) -> None:
    
    jobs = {}

    out_dir = os.path.dirname(args.out_file)

    log_folder = os.path.join(out_dir, "log_test/")

    for dataset in datasets:
            
        print(f"Running {dataset.name}")
        # print("Downloading dataset")
        X, y, categorical_indicator, dataset_id = dataset.X, dataset.y, dataset.categorical_columns, dataset.id

        try:
            # give atleast 1 min per split and 1.5 times the opt_time
            total_job_time = args.slurm_job_time #max(time * 1.5, 120) * args.chunk_size
            slurm_executer = get_executer(
                partition=args.partition,
                log_folder=log_folder,
                total_job_time_secs=total_job_time,
                gpu=device!="cpu")
            res_dic = slurm_executer.submit(run_on_dataset(
                X=X,
                y=y,
                categorical_indicator=categorical_indicator,
                args=args,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                resnet_config=resnet_config
            ))

        except:
            print("FAILED")
            print(traceback.format_exc())
            res_dic = {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "original_n_samples": pd.NA,
                "original_n_features": pd.NA,
                "num_categorical_columns": pd.NA,
                "num_pseudo_categorical_columns": pd.NA,
                "num_columns_missing": pd.NA,
                "num_rows_missing": pd.NA,
                "too_easy": pd.NA,
                "score_resnet": pd.NA,
                "score_linear": pd.NA,
                "score_hgbt": pd.NA,
                "score_tree": pd.NA,
                "score_mlp": pd.NA,
                "score_rf": pd.NA,
                "heterogeneous": pd.NA,
                "n_samples": pd.NA,
                "too_small": pd.NA}


        jobs[dataset_id] = res_dic

    res_df = pd.DataFrame()

    for dataset_id in jobs:
        if hasattr(jobs[dataset_id], 'result'):
            print(f"Waiting for result on : {dataset_id}, with job_id: {jobs[dataset_id].job_id}")
            res_dic = jobs[dataset_id].result()
            print(f"Job finished for {dataset_id}, with job_id: {jobs[dataset_id].job_id}")
        else:
            res_dic = jobs[dataset_id]
        res_df = res_df.append(res_dic, ignore_index=True)

    res_df.to_csv("{}.csv".format(args.out_file))


def do_evaluations(
    datasets: list[Dataset],
    args: argparse.Namespace,
    resnet_config: dict,
    device: str = "cpu",
    only_fetch_data: bool = False
) -> None:
    res_df = pd.DataFrame()

    for dataset in datasets:
            
        print(f"Running {dataset.name}")
        # print("Downloading dataset")
        X, y, categorical_indicator, dataset_id = dataset.X, dataset.y, dataset.categorical_columns, dataset.id
        if only_fetch_data:
            continue
        try:
            res_dic = run_on_dataset(
                X=X,
                y=y,
                categorical_indicator=categorical_indicator,
                args=args,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                resnet_config=resnet_config
            )
            res_df = res_df.append(res_dic, ignore_index=True)
            res_df.to_csv("{}.csv".format(args.out_file))

        except:
            print("FAILED")
            print(traceback.format_exc())
            pass
            res_dic = {
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "original_n_samples": pd.NA,
                "original_n_features": pd.NA,
                "num_categorical_columns": pd.NA,
                "num_pseudo_categorical_columns": pd.NA,
                "num_columns_missing": pd.NA,
                "num_rows_missing": pd.NA,
                "too_easy": pd.NA,
                "score_resnet": pd.NA,
                "score_linear": pd.NA,
                "score_hgbt": pd.NA,
                "score_tree": pd.NA,
                "score_mlp": pd.NA,
                "score_rf": pd.NA,
                "heterogeneous": pd.NA,
                "n_samples": pd.NA,
                "too_small": pd.NA}

        res_df = res_df.append(res_dic, ignore_index=True)

    if not only_fetch_data:
        res_df.to_csv("{}.csv".format(args.out_file))

if __name__ == '__main__':
    args = arguments()
    print(args)
    if args.datasets is None:
        valid_datasets = Dataset.fetch("too_easy")
        test_datasets = Dataset.fetch("benchmark_dids")
        all_datasets = valid_datasets + test_datasets
    else:
        all_datasets = Dataset.fetch(args.datasets)

    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() and not args.device == "cpu" else 'cpu'
    print(device)

    if args.remove_model is None:
        args.remove_model = []
    
    resnet_config = {"model_type": "skorch",
                    "model__use_checkpoints": True,
                    "model__optimizer": "adamw",
                    "model__lr_scheduler": True,
                    "model__batch_size": 512,
                    "model__max_epochs": 300,
                    "model__module__activation": "reglu",
                    "model__module__normalization": "batchnorm",
                    "model__module__n_layers": 8,
                    "model__module__d": 256,
                    "model__module__d_hidden_factor": 2,
                    "model__module__hidden_dropout": 0.2,
                    "model__module__residual_dropout": 0.2,
                    "model__lr": 1e-3,
                    "model__optimizer__weight_decay": 1e-7,
                    "model__module__d_embedding": 128,
                    "model__verbose": 100,
                    "model__device": device}

    if args.regression:
        resnet_config["model_name"] = "rtdl_resnet_regressor"
        resnet_config["regression"] = True
        resnet_config["data__regression"] = True
        resnet_config["transformed_target"] = True
    else:
        resnet_config["model_name"] = "rtdl_resnet"
        resnet_config["regression"] = False
        resnet_config["data__regression"] = False
        resnet_config["transformed_target"] = False

    if args.categorical:
        resnet_config["data__categorical"] = True
    else:
        resnet_config["data__categorical"] = False

    if args.parallel:
        do_evaluations_parallel(
            datasets=all_datasets,
            args=args,
            resnet_config=resnet_config,
            device=device,
            only_fetch_data=args.only_fetch_data)
    else:
        do_evaluations(
            datasets=all_datasets,
            args=args,
            resnet_config=resnet_config,
            device=device,
            only_fetch_data=args.only_fetch_data)
