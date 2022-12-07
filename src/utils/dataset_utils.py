from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import pandas as pd
import numpy as np

import openml

from generate_dataset_pipeline import generate_dataset

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

@dataclass
class Dataset:
    """Small helper class just to name entries in the loaded pickled datasets."""

    name: str
    X_train: np.ndarray | pd.DataFrame
    y_train: np.ndarray | pd.DataFrame
    X_valid: np.ndarray | pd.DataFrame
    y_valid: np.ndarray | pd.DataFrame
    X_test: np.ndarray | pd.DataFrame
    y_test: np.ndarray | pd.DataFrame
    categorical_indicator: list[bool]
    # Seems to be some things about how the dataset was constructed
    id: int


    @property
    def categorical(self) -> bool:
        return sum(self.categorical_indicator) == self.X_train.shape[1]

    @property
    def numerical(self) -> bool:
        return sum(self.categorical_indicator) == 0

    @property
    def mixed(self) -> bool:
        return not self.numerical and not self.categorical

    @classmethod
    def fetch(
        self,
        dataset_id: int,
        **config
    ) -> Dataset:
        dataset = Dataset.from_openml(dataset_id, config)
        return dataset

    @classmethod
    def from_openml(
        self,
        dataset_id: int,
        seed: int,
        dataset_name: str,
        **config
    ) -> list[Dataset]:
        config.update({'data__keyword': dataset_id})
        X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(seed))

        return Dataset(
            dataset_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            categorical_indicator=categorical_indicator,
            id=dataset_id)

    def as_list(self) -> list:
        """How the internals expect a dataset to look like."""
        return [
            self.name,
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.X_test,
            self.y_test,
            self.categorical_indicator,
            self.id
        ]
