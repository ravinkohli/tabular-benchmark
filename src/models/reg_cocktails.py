
from __future__ import annotations
import json
import os
import tempfile as tmp
import warnings
from typing import Any
import logging.handlers
import time

import pandas as pd
import numpy as np

from ConfigSpace import Configuration

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline # TabularClassificationTask
from autoPyTorch.datasets.tabular_dataset import TabularDataset
cocktails = False
try:
    from autoPyTorch.automl_common.common.utils.backend import Backend, create
except ModuleNotFoundError:
    cocktails = True
    from autoPyTorch.utils.backend import Backend, create
from autoPyTorch.utils.pipeline import get_dataset_requirements
from autoPyTorch.utils.common import replace_string_bool_to_bool
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics
# from autoPyTorch.utils.logging_ import setup_logger, get_named_client_logger, start_log_server



def get_updates_for_regularization_cocktails(preprocess: bool = False):
    """
    These updates replicate the regularization cocktail paper search space.
    Args:
    Returns:
    ________
        search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

    include_updates = dict()
    include_updates['network_embedding'] = ['NoEmbedding']
    include_updates['network_init'] = ['NoInit']

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

    print('has numerical features')
    search_space_updates.append(
        node_name='imputer',
        hyperparameter='numerical_strategy',
        value_range=['median'],
        default_value='median',
    )
    if not preprocess:
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
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

    return search_space_updates, include_updates


def get_updates_for_autopytorch_tabular(preprocess: bool = False):
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
    # from autoPyTorch.constants import MIN_CATEGORIES_FOR_EMBEDDING_MAX


    """
    These updates mimic the autopytorch tabular paper.
    Returns:
    ________
    search_space_updates - HyperparameterSearchSpaceUpdates
        The search space updates like setting different hps to different values or ranges.
    """

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

    # weights initialisation
    search_space_updates.append(
        node_name='network_init',
        hyperparameter='__choice__',
        value_range=['NoInit'],
        default_value='NoInit',
    )
    search_space_updates.append(
        node_name='network_init',
        hyperparameter='NoInit:bias_strategy',
        value_range=['Zero'],
        default_value='Zero',
    )

    # backbone architecture choices
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone', 'ShapedMLPBackbone'],
        default_value='ShapedResNetBackbone',
    )

    # resnet backbone
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:dropout_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_dropout',
        value_range=[0, 1],
        default_value=0.5,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[1, 4],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[1, 3],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[32, 512],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[32, 512],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:use_skip_connection',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:use_batch_norm',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['shake-shake'],
        default_value='shake-shake',
    )
    # mlp backbone
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:mlp_shape',
        value_range=['funnel'],
        default_value='funnel',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:num_groups',
        value_range=[1, 5],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:output_dim',
        value_range=[64, 1024],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:max_units',
        value_range=[64, 1024],
        default_value=64,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedMLPBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # training updates
    # lr scheduler
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingLR'],
        default_value='CosineAnnealingLR',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingLR:T_max',
        value_range=[50],
        default_value=50,
    )
    # optimizer
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamOptimizer', 'SGDOptimizer'],
        default_value='AdamOptimizer',
    )
    # adam
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:lr',
        value_range=[1e-4, 1e-1],
        default_value=1e-3,
        log=True
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:use_weight_decay',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:weight_decay',
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )

    # sgd
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:lr',
        value_range=[1e-4, 1e-1],
        default_value=1e-3,
        log=True
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:use_weight_decay',
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:weight_decay',
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='SGDOptimizer:momentum',
        value_range=[0.1, 0.999],
        default_value=0.1,
        log=True
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[16, 512],
        default_value=128,
        log=True
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor', 'TruncatedSVD'],
        default_value='NoFeaturePreprocessor',
    )
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='TruncatedSVD:target_dim',
        value_range=[0.1, 0.9],
        default_value=0.4,
    )
    search_space_updates.append(
        node_name='imputer',
        hyperparameter='numerical_strategy',
        value_range=['mean'],
        default_value='mean',
    )
    if not preprocess:
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )
    # trainer
    trainer_choices = ['StandardTrainer', 'MixUpTrainer']
    search_space_updates.append(
        node_name='trainer',
        hyperparameter='__choice__',
        value_range=trainer_choices,
        default_value=trainer_choices[0],
    )
    for trainer_choice in trainer_choices:
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_lookahead_optimizer',
            value_range=[False],
            default_value=False,
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_snapshot_ensemble',
            value_range=[False],
            default_value=False,
        )
        search_space_updates.append(
            node_name='trainer',
            hyperparameter=f'{trainer_choice}:use_stochastic_weight_averaging',
            value_range=[False],
            default_value=False,
        )
    return search_space_updates, None


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
    dataset: TabularDataset,
    validator: TabularInputValidator,
    seed: int,
    budget: int,
    backend: Backend,
    autopytorch_source_dir: str,
    dataset_properties: dict[str, Any],
    num_run: int = 0,
    device: str = "cpu",
    X_test: None | pd.DataFrame | np.ndarray = None,
    y_test: None | pd.DataFrame | np.ndarray = None,
    metric_name: str = "accuracy",
    configuration: None | Configuration = None,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    preprocess: bool = False,
    skip_training: bool = False
) -> float:
    ############################################################################
    # Build and fit a classifier
    # ==========================

    start_time = time.time()
    if not skip_training:
        print(f"Using preprocess: {preprocess}")
        search_space_updates, include_updates = get_updates_for_regularization_cocktails(preprocess) if cocktails else get_updates_for_autopytorch_tabular(preprocess)
        backend.save_datamanager(dataset)
        pipeline_options = replace_string_bool_to_bool(json.load(open(
                    os.path.join(autopytorch_source_dir, 'autoPyTorch/configs/default_pipeline_options.json'))))
        pipeline_options.update( # No early stopping and train on gpu
        {
            'early_stopping': -1,
            'min_epochs': budget,
            'epochs': budget,
        })
        pipeline_options.update({'device': device})
        pipeline = TabularClassificationPipeline(
            dataset_properties=dataset_properties,
            random_state=seed,
            include=include_updates,
            search_space_updates=search_space_updates,
            config=configuration,
        )


        fit_dictionary = init_fit_dictionary(
            pipeline_config=pipeline_options,
            dataset_properties=dataset_properties,
            budget_type="epochs",
            budget=budget,
            metric_name=metric_name,
            dataset=dataset,
            backend=backend,
            num_run=num_run,
            logger_port=logger_port
        )

        # print(repr(pipeline.get_hyperparameter_search_space()))
        ############################################################################
        # Search for an ensemble of machine learning algorithms
        # =====================================================
        pipeline.fit(X=fit_dictionary)
    else:
        import pickle
        pipeline = pickle.load(open(os.path.join(backend.get_numrun_directory(seed=seed, num_run=num_run, budget=budget), backend.get_model_filename(seed=seed, idx=num_run, budget=budget)), "rb"))
    ############################################################################
    # Print the final ensemble performance
    # ====================================
    metrics = get_metrics(dataset_properties=dataset_properties, names=[metric_name])

    y_train_pred = pipeline.predict(dataset.train_tensors[0], batch_size=512)
    train_score = calculate_score(dataset.train_tensors[1], y_train_pred, metrics=metrics, task_type=dataset_properties.get("task_type"))[metric_name]
    y_val_pred = None
    if dataset.test_tensors[0] is not None:
        y_val_pred = pipeline.predict(dataset.test_tensors[0], batch_size=512)
        val_score = calculate_score(dataset.test_tensors[1], y_val_pred, metrics=metrics, task_type=dataset_properties.get("task_type"))[metric_name]
    else:
        val_score = 0
    if X_test is not None:
        X_test_transformed, y_test_transformed = validator.transform(X_test, y_test)
        y_test_pred = pipeline.predict(X_test_transformed, batch_size=512)
        test_score = calculate_score(y_test_transformed, y_test_pred, metrics=metrics, task_type=dataset_properties.get("task_type"))[metric_name]
    else:
        test_score = val_score

    backend.save_numrun_to_dir(seed=seed, idx=num_run, budget=budget, model=pipeline, test_predictions=y_val_pred, cv_model=None, ensemble_predictions=None, valid_predictions=None)
    return {
        'train': train_score,
        'test': test_score,
        'val': val_score,
        'duration': time.time() - start_time
    }

