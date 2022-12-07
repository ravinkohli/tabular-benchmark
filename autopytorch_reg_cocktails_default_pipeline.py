"""
======================
Tabular Classification
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
from __future__ import annotations
import json
import os
import tempfile as tmp
import warnings
from typing import Any
import logging.handlers
import argparse
import pickle
import pandas as pd
import numpy as np

import time

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import openml
import sklearn.model_selection

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline # TabularClassificationTask
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
cocktails = False
try:
    from autoPyTorch.automl_common.common.utils.backend import Backend, create
except ModuleNotFoundError:
    cocktails = True
    from autoPyTorch.utils.backend import Backend, create
from autoPyTorch.utils.pipeline import get_dataset_requirements, get_configuration_space
from autoPyTorch.utils.common import replace_string_bool_to_bool
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics



def get_updates_for_regularization_cocktails(
    categorical_indicator: np.ndarray):
    """
    These updates replicate the regularization cocktail paper search space.
    Args:
        categorical_indicator (np.ndarray)
            An array that indicates whether a feature is categorical or not.
    Returns:
    ________
        pipeline_update, search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The pipeline updates like number of epochs, budget, seed etc.
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

    include_updates = dict()
    include_updates['network_embedding'] = ['NoEmbedding']
    include_updates['network_init'] = ['NoInit']

    has_cat_features = any(categorical_indicator)
    has_numerical_features = not all(categorical_indicator)

    def str2bool(v):
        if isinstance(v, bool):
            return [v, ]
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return [True, ]
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return [False, ]
        elif v.lower() == 'conditional':
            return [True, False]
        else:
            raise ValueError('No valid value given.')
    search_space_updates = HyperparameterSearchSpaceUpdates()

    # architecture head
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='__choice__',
        value_range=['no_head'],
        default_value='no_head',
    )
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='no_head:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # backbone architecture
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone'],
        default_value='ShapedResNetBackbone',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['brick'],
        default_value='brick',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['even-even'],
        default_value='even-even',
    )

    # training updates
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingWarmRestarts'],
        default_value='CosineAnnealingWarmRestarts',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingWarmRestarts:n_restarts',
        value_range=[3],
        default_value=3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamWOptimizer'],
        default_value='AdamWOptimizer',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:lr',
        value_range=[1e-3],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[128],
        default_value=128,
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor'],
        default_value='NoFeaturePreprocessor',
    )

    if has_numerical_features:
        print('has numerical features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['median'],
            default_value='median',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )

    if has_cat_features:
        print('has cat features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='categorical_strategy',
            value_range=['constant_!missing!'],
            default_value='constant_!missing!',
        )
        search_space_updates.append(
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder'],
            default_value='OneHotEncoder',
        )

    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )

    # if the cash formulation of the cocktail is not activated,
    # otherwise the methods activation will be chosen by the SMBO optimizer.


    # No early stopping and train on gpu
    pipeline_update = {
        'early_stopping': -1,
        'min_epochs': 105,
        'epochs': 105,
        "device": 'cpu',
    }

    return pipeline_update, search_space_updates, include_updates

def init_fit_dictionary(
    pipeline_config: dict[str, Any],
    dataset_properties: dict[str, Any],
    budget_type: str,
    budget: float,
    metric_name: str,
    backend: Backend,
    dataset: TabularDataset,
    num_run: int = 0,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    metrics_dict: dict[str, list[str]] | None = None,
) -> None:
    """
    Initialises the fit dictionary

    Args:
        logger_port (int):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel.
        pipeline_config (Dict[str, Any]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        metrics_dict (Optional[Dict[str, List[str]]]):
        Contains a list of metric names to be evaluated in Trainer with key `additional_metrics`. Defaults to None.

    Returns:
        None
    """

    fit_dictionary: dict[str, Any] = {'dataset_properties': dataset_properties}

    if metrics_dict is not None:
        fit_dictionary.update(metrics_dict)

    split_id = 0
    train_indices, val_indices = dataset.splits[split_id]

    fit_dictionary.update({
        'X_train': dataset.train_tensors[0],
        'y_train': dataset.train_tensors[1],
        'X_test': dataset.test_tensors[0],
        'y_test': dataset.test_tensors[1],
        'backend': backend,
        'logger_port': logger_port,
        'optimize_metric': metric_name,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'split_id': split_id,
        'num_run': num_run
    })

    fit_dictionary.update(pipeline_config)
    # If the budget is epochs, we want to limit that in the fit dictionary
    if budget_type == 'epochs':
        fit_dictionary['epochs'] = budget
        fit_dictionary.pop('runtime', None)
    elif budget_type == 'runtime':
        fit_dictionary['runtime'] = budget
        fit_dictionary.pop('epochs', None)
    else:
        raise ValueError(f"budget type must be `epochs` or `runtime`"
                            f"(Only used by forecasting taskss), but got {budget_type}")
    return fit_dictionary


def run_on_autopytorch(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.DataFrame | np.ndarray,
    seed: int,
    budget: int,
    out_dir: str,
    autopytorch_source_dir: str,
    categorical_indicator: np.ndarray,
    num_run: int = 0,
    device: str = "cpu",
    metric_name: str = "balanced_accuracy",
    is_classification: bool = True
) -> float:
    ############################################################################
    # Build and fit a classifier
    # ==========================
    temp_dir = os.path.join(out_dir, "tmp_1")
    out_dir = os.path.join(out_dir, "out_1")
    backend_kwargs = dict(
        temporary_directory=temp_dir,
        output_directory=out_dir,
        delete_output_folder_after_terminate=False,
        delete_tmp_folder_after_terminate=False
    )

    if not cocktails:
        backend_kwargs.update({'prefix':'autopytorch_pipeline'})
    backend = create(
        **backend_kwargs
    )
    backend.setup_logger(port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, name="autopytorch_pipeline")
    validator_kwargs = dict(is_classification=is_classification)
    if not cocktails:
        validator_kwargs.update({'seed': seed})
    validator = TabularInputValidator(
        **validator_kwargs)
    validator = validator.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        validator=validator,
        seed=seed
    )
    dataset_requirements = get_dataset_requirements(
                info=dataset.get_required_dataset_info(),
                # include=include_components,
                # exclude=exclude_components,
                # search_space_updates=search_space_updates
        )
    dataset_properties = dataset.get_dataset_properties(dataset_requirements)
    backend.save_datamanager(dataset)
    pipeline_options = replace_string_bool_to_bool(json.load(open(
                os.path.join(autopytorch_source_dir, 'autoPyTorch/configs/default_pipeline_options.json'))))
    pipeline_update, search_space_updates, include_updates = get_updates_for_regularization_cocktails(categorical_indicator=categorical_indicator)
    pipeline_options.update(pipeline_update)
    pipeline_options.update({'device': device})
    pipeline = TabularClassificationPipeline(
        dataset_properties=dataset_properties,
        random_state=seed,
        include=include_updates,
        search_space_updates=search_space_updates
    )

    print(pipeline.configuration)
    print('\n' + repr(pipeline.configuration_space))
    fit_dictionary = init_fit_dictionary(
        pipeline_config=pipeline_options,
        dataset_properties=dataset_properties,
        budget_type="epochs",
        budget=budget,
        metric_name=metric_name,
        dataset=dataset,
        backend=backend,
        num_run=num_run
    )

    # print(repr(pipeline.get_hyperparameter_search_space()))
    ############################################################################
    # Search for an ensemble of machine learning algorithms
    # =====================================================
    pipeline.fit(X=fit_dictionary)

    ############################################################################
    # Print the final ensemble performance
    # ====================================
    y_pred = pipeline.predict(dataset.test_tensors[0])

    metrics = get_metrics(dataset_properties=dataset_properties, names=[metric_name])
    score = calculate_score(dataset.test_tensors[1], y_pred, metrics=metrics, task_type=dataset_properties.get("task_type"))

    backend.save_numrun_to_dir(seed=seed, idx=num_run, budget=BUDGET, model=pipeline, test_predictions=y_pred, cv_model=None, ensemble_predictions=None, valid_predictions=None)
    return list(score.values())[0]

############################################################################
# Data Loading
# ============

parser = argparse.ArgumentParser(
    "Run autopytorch pipeline"
)
parser.add_argument(
    '--device',
    type=str,
    default="cpu"
)
parser.add_argument(
    '--dataset_id',
    type=int,
    default=40981
)
parser.add_argument(
    '--autopytorch_source_dir',
    type=str,
    default="/home/rkohli/tabular-benchmark/apt_reg_cocktails/Auto-PyTorch"
)

if __name__ == '__main__':
    args = parser.parse_args()
    SEED = 1
    BUDGET = 105
    dataset = openml.datasets.get_dataset(args.dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=SEED,
    )

    start_time = time.time()
    final_score = run_on_autopytorch(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=SEED,
        budget=BUDGET,
        num_run=0,
        device=args.device,
        autopytorch_source_dir=args.autopytorch_source_dir,
        out_dir=f'./tmp/{dataset.name}',
        categorical_indicator=categorical_indicator
    )
    print(f"Finsihed training with score:{final_score} in {time.time()-start_time}")
    