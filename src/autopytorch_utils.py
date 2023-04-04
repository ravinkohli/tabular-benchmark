import openml
from sklearn.model_selection import train_test_split
import numpy as np

from preprocessing.preprocessing import preprocessing


def get_data_from_openml_task_id(
        task_id: int,
        val_share: float = 0.33,
):
    """
    Args:
    _____
        task_id: int
            The id of the task which will be used for the run.
        val_share: float
            The validation split size from the train set.
        test_size: float
            The test split size from the whole dataset.
        seed: int
            The seed used for the dataset preparation.
    Returns:
    ________
    X_train, X_test, y_train, y_test, resampling_strategy_args, categorical indicator: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict, np.ndarray]
        The train examples, the test examples, the train labels, the test labels, the resampling strategy to be used and the categorical indicator for the features.
    """
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )

    train_indices, test_indices = task.get_train_test_split_indices()
    # AutoPyTorch fails when it is given a y DataFrame with False and True
    # values and category as dtype. in its inner workings it uses sklearn
    # which cannot detect the column type.
    if isinstance(y[1], bool):
        y = y.astype('bool')

    # uncomment only for np.arrays

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test, categorical_indicator, dataset


def get_preprocessed_openml_dataset(
        dataset_id,
        categorical=False,
        regression=False,
        transformation=None,
        remove_pseudo_categorical=True,
        remove_high_cardinality_columns=True,
        remove_columns_with_nans=True,
        balance_classes=True
    ):
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    print("Downloading data")
    print(dataset.name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
    num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
        preprocessing(
            X, y, categorical_indicator, categorical=categorical,
            regression=regression, transformation=transformation,
            dataset_id=dataset_id, remove_pseudo_categorical=remove_pseudo_categorical,
            remove_high_cardinality_columns=remove_high_cardinality_columns,
            remove_columns_with_nans=remove_columns_with_nans,
            balance_classes=balance_classes)
    try:
        X = X.sparse.to_dense()
    except:
        X = X
    return X, y, categorical_indicator

def get_openml_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
    print("Downloading data")
    print(dataset.name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    try:
        X = X.sparse.to_dense()
    except:
        X = X
    return X, y, categorical_indicator

def get_train_test_split(X, y, train_prop = 0.7, max_train_samples=10_000, iter=1):
    if max_train_samples is not None:
        train_prop = min(max_train_samples / X.shape[0], train_prop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop,
                                                        random_state=np.random.RandomState(iter))
    return X_train, X_test, y_train, y_test

def data_to_train_test_valid(x, y, config, rng=None):
    n_rows = x.shape[0]
    if not config.get("max_train_samples", None) is None:
        train_set_prop = min(config["max_train_samples"] / n_rows, config["train_prop"])
    else:
        train_set_prop = config["train_prop"]
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, train_size=train_set_prop, random_state=rng)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size= config["val_test_prop"], random_state=rng)
    if not config.get("max_val_samples", None) is None and x_val.shape[0] > config["max_val_samples"]:
        x_val = x_val[:config["max_val_samples"]]
        y_val = y_val[:config["max_val_samples"]]
    if not config.get("max_test_samples", None) is None and x_test.shape[0] > config["max_test_samples"]:
        x_test = x_test[:config["max_test_samples"]]
        y_test = y_test[:config["max_test_samples"]]
    return x_train, x_val, x_test, y_train, y_val, y_test