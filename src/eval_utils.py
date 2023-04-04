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



benchmark_dids = dict(
    numerical = [
        44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131
    ],
    categorical=[
        44156, 44157, 44159, 44160, 44161, 44162, 44186
    ]
)
too_easy_dids = dict(
    numerical = [
        44, 152, 153, 246, 251, 256, 257, 258, 267, 269, 351, 357, 720, 725, 734, 735, 737, 761, 803, 816, 819, 823, 833, 846, 847, 871, 976, 979, 1053, 1119, 1181, 1205, 1212, 1216, 1218, 1219, 1240, 1241, 1242, 1486, 1507, 1590, 4134, 23517, 41146, 41147, 41162, 42206, 42343, 42395, 42435, 42477, 42742, 42750, 43489, 60, 150, 159, 160, 180, #]
        182, 250, 252, 254, 261, 266, 271, 279, 554, 1110, 1113, 1183, 1185, 1209, 1214, 1222, 1226, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1393, 1394, 1395, 1476, 1477, 1478, 1503, 1526, 1596, 4541, 40685, 40923, 40996, 40997, 41000, 41002, 41039, 41163, 41164, 41166, 41168, 41169, 41671, 41972, 41982, 41986, 41988, 41989, 42468, 42746
         ],
    categorical=[
        4, 26, 154, 179, 274, 350, 720, 881, 923, 959, 981, 993, 1110, 1112, 1113, 1119, 1169, 1240, 1461, 1486, 1503, 1568, 1590, 4534, 4541, 40517, 40672, 40997, 40998, 41000, 41002, 41003, 41006, 41147, 41162, 41440, 41672, 42132, 42192, 42193, 42206, 42343, 42344, 42345, 42477, 42493, 42732, 42734, 42742, 42746, 42750, 43044, 43439, 43489, 43607, 43890, 43892, 43898, 43903, 43904, 43920, 43922, 43923, 43938
        ]
    )

not_too_easy_dids = dict(
    numerical = [
        # 41081
        6, 1037, 1039, 1040, 1044, 1557, 1558, 40983, 28, 30, 43551, 1056, 32, 38, 41004, 1069, 40498, 40499, 4154, 1597, 43072, 41027, 1111, 40536, 1112, 1114, 1116, 1120, 41082, 41103, 151, 41138, 183, 41150, 41156, 41160, 41161, 41165, 42192, 42193, 722, 727, 728, 40666, 42733, 40701, 42757, 42758, 42759, 40713, 41228, 42252, 42256, 273, 42769, 42773, 42774, 42775, 293, 300, 821, 310, 350, 354, 375, 390, 399, 42397, 1459, 1461, 1475, 40900, 40910, 1489, 977, 980, 41434, 41946, 993, 1000, 1002, 1018, 1019, 1021
    ]
)

remaining_dids = dict(
    numerical = [
        1120, 993, 354, 293, 41160, 42769, 722, 1489, 1044, 821, 1461, 151, 1112, 1114, 41150,
        6, 28, 30, 32, 38, 183, 273, 300, 310, 350, 375, 390, 399, 727, 728, 977, 980, 1000, 1002, 1018, 1019, 1021, 1037, 1039, 1040, 1056, 1069, 1111, 1116, 1459, 1475, 1557, 1558, 1597, 4154, 40498, 40499, 40536, 40666, 40701, 40713, 40900, 40910, 40983, 41004, 41027, 41081, 41082, 41103, 41138, 41156, 41161, 41165, 41228, 41434, 41946, 42192, 42193, 42252, 42256, 42397, 42733, 42757, 42758, 42759, 42773, 42774, 42775, 43072, 43551
    ]
)

PREDEFINED_DATASET_COLLECTIONS = {
    'too_easy': too_easy_dids,
    "benchmark": benchmark_dids,
    "not_too_easy": not_too_easy_dids,
    "remaining":remaining_dids
}


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


def get_executer_params(timeout: float, partition: str, gpu: bool = False, cpu_nodes: int = 1, mem_per_cpu: int = 2000) -> Dict[str, Any]:
    if gpu:
        if "bosch" in partition:
            return {'time': int(timeout), 'partition': partition, 'gres': f"gpu:{cpu_nodes}"} #, 'gpus': cpu_nodes} #, 'nodes': 1} #, 'cpus_per_task': cpu_nodes}
        else:
            return {'timeout_min': int(timeout), 'slurm_partition': partition, 'slurm_gres': f"gpu:{cpu_nodes}"} #, 'slurm_num_gpus': cpu_nodes} #slurm_gpus_per_task': cpu_nodes} #  'slurm_tasks_per_node': 1,
    else:
        return {'time': int(timeout), 'partition': partition, 'mem_per_cpu': mem_per_cpu, 'nodes': 1, 'cpus_per_task': cpu_nodes}


def get_executer(partition: str, log_folder: str, gpu: bool=False, total_job_time_secs: float = 3600, cpu_nodes: int = 1, mem_per_cpu: int = 2000):
    slurm_executer = get_executer_class(partition)(folder=log_folder)
    slurm_executer.update_parameters(**get_executer_params(np.ceil(total_job_time_secs/60), partition, gpu, cpu_nodes=cpu_nodes, mem_per_cpu=mem_per_cpu))
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

