from __future__ import annotations

import argparse
import pickle
import re
from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import chain, product
from pathlib import Path
import random
from typing import Any, Callable, Optional, Tuple, Dict, Iterable, Sequence
import os

import numpy as np

import pandas as pd

import torch

import openml
                                        
from submitit import SlurmExecutor, AutoExecutor



class BoschSlurmExecutor(SlurmExecutor):
    def _make_submission_command(self, submission_file_path):
        return ["sbatch", str(submission_file_path), '--bosch']


PARTITION_TO_EXECUTER = {
    'bosch': BoschSlurmExecutor,
    'other': AutoExecutor

}

def get_openml_classification(did):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    return X, y, categorical_indicator, attribute_names

def load_openml_list(dids,
                     subsample_flag: bool = False):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")

    for ds in datalist.index:
        # If we are going to subsample in the splits, these will all be capped
        modifications = {
            'samples_capped': subsample_flag,
            'classes_capped': subsample_flag,
            'feats_capped': subsample_flag,
        }
        entry = datalist.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported")
            #X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            try:
                X, y, categorical_feats, attribute_names = get_openml_classification(int(entry.did))
            except Exception as e:
                print(f"Couldn't load data for {entry.did}")
                continue
        datasets += [[entry['name'], X, y, categorical_feats, attribute_names, modifications, int(entry.did)]]

    return datasets, datalist

def get_executer_class(partition: str) -> SlurmExecutor:
    if 'bosch' in partition:
        key = 'bosch'
    else:
        key = 'other'
    return PARTITION_TO_EXECUTER[key]


def get_executer_params(timeout: float, partition: str, gpu: bool = False) -> Dict[str, Any]:
    if gpu:
        return {'timeout_min': int(timeout), 'slurm_partition': partition, 'slurm_gres': "gpu:1"} #  'slurm_tasks_per_node': 1,
    else:
        return {'time': int(timeout), 'partition': partition, 'mem_per_cpu': 12000, 'nodes': 1, 'cpus_per_task': 1, 'ntasks_per_node': 1}


def get_executer(partition: str, log_folder: str, gpu: bool=False, total_job_time_secs: float = 3600):
    slurm_executer = get_executer_class(partition)(folder=log_folder)
    slurm_executer.update_parameters(**get_executer_params(np.ceil(total_job_time_secs/60), partition, gpu))
    return slurm_executer


def set_seed(seed: int):
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@dataclass
class Dataset:
    """Small helper class just to name entries in the loaded pickled datasets."""

    name: str
    X: torch.Tensor
    y: torch.Tensor
    categorical_columns: list[int]
    attribute_names: list[str]
    # Seems to be some things about how the dataset was constructed
    info: dict
    id: int
    # Only 'multiclass' is known?
    task_type: str

    @property
    def categorical(self) -> bool:
        return len(self.categorical_columns) == len(self.attribute_names)

    @property
    def numerical(self) -> bool:
        return len(self.categorical_columns) == 0

    @property
    def mixed(self) -> bool:
        return not self.numerical and not self.categorical

    @classmethod
    def fetch(
        self,
        identifier: str | int | list[int],
        use_categorical_predefined: bool = False,
        only: Callable | None = None,
    ) -> list[Dataset]:
        if isinstance(identifier, str) and identifier in PREDEFINED_DATASET_COLLECTIONS:
            subset = "numerical" if not use_categorical_predefined else "categorical"
            dids = PREDEFINED_DATASET_COLLECTIONS[identifier][subset]
            datasets = Dataset.from_openml(dids)
        elif isinstance(identifier, int):
            identifier = [identifier]
            datasets = Dataset.from_openml(identifier)
        elif isinstance(identifier, list):
            datasets = Dataset.from_openml(identifier)
        else:
            raise ValueError(identifier)

        if only:
            return list(filter(only, datasets))
        else:
            return datasets

    @classmethod
    def from_openml(
        self,
        dataset_id: int | list[int],
        multiclass: bool = True,
    ) -> list[Dataset]:
        # TODO: should be parametrized, defaults taken from ipy notebook
        if not isinstance(dataset_id, list):
            dataset_id = [dataset_id]

        datasets, _ = load_openml_list(
            dataset_id,
        )
        return [
            Dataset(  # type: ignore
                *entry,
                task_type="multiclass" if multiclass else "binary",
            )
            for entry in datasets
        ]

    def as_list(self) -> list:
        """How the internals expect a dataset to look like."""
        return [
            self.name,
            self.X,
            self.y,
            self.categorical_columns,
            self.attribute_names,
            self.info,
            self.id
        ]


@dataclass
class Results:
    # Big ass predefined dictionary
    df: pd.DataFrame

    def at(
        self,
        *,
        method: str | list[str] | None = None,
        seed: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ) -> Results:
        """Use this for slicing in to the dataframe to get what you need"""
        df = self.df
        items = {
            "method": method,
            "seed": seed,
        }
        for name, item in items.items():
            if item is None:
                continue
            idx: list = item if isinstance(item, list) else [item]
            df = df[df.index.get_level_values(name).isin(idx)]
            if not isinstance(item, list):
                df = df.droplevel(name, axis="index")

        if dataset:
            _dataset = dataset if isinstance(dataset, list) else [dataset]
            df = df.T.loc[df.T.index.get_level_values("dataset").isin(_dataset)].T
            if not isinstance(dataset, list):
                df = df.droplevel("dataset", axis="columns")

        if metric:
            _metric = metric if isinstance(metric, list) else [metric]
            df = df.T.loc[df.T.index.get_level_values("metric").isin(_metric)].T
            if not isinstance(metric, list):
                df = df.droplevel("metric", axis="columns")

        return Results(df)

    @property
    def methods(self) -> list[str]:
        return list(self.df.index.get_level_values("method").unique())

    @property
    def datasets(self) -> list[str]:
        return list(self.df.columns.get_level_values("dataset").unique())

    @property
    def metrics(self) -> list[str]:
        return list(self.df.columns.get_level_values("metric").unique())


def arguments() -> argparse.Namespace:
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    # Add the arguments
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--slurm_job_time', type=int, default=60, help='Time on slurm in minutes')
    parser.add_argument('--file', type=str, default="filename", help='Csv with all datasets')
    parser.add_argument('--out_file', type=str, default="filename", help='filename to save')
    parser.add_argument('--regression', action='store_true', help='True if regression, false otherwise')
    parser.add_argument('--parallel', action='store_true', help='True if run parallely on slurm, false otherwise')
    parser.add_argument('--only_fetch_data', action='store_true', help='True if only fetch data, false otherwise')
    parser.add_argument('--categorical', action='store_true')
    parser.add_argument('--all', action='store_true', help="Whether to check all datasets or only those already "
                                                        "deemed too easy with a HGBT")
    parser.add_argument('--remove_model', nargs='+', help='List of models not to try')
    parser.add_argument('--datasets', nargs='+', default=None, help='List of models not to try')
    parser.add_argument(
        "--partition", type=str, default="bosch_cpu-cascadelake"
    )
    args = parser.parse_args()

    # Parse args.datasets manually as it could be a str or list of int
    if args.datasets is not None:
        try:
            datasets = [int(i) for i in args.datasets]
        except Exception:
            assert len(args.datasets) == 1, args
            datasets = args.datasets[0]

        args.datasets = datasets

    # Parse the arguments
    return args

