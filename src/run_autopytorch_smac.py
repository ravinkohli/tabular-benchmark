import argparse
import os
from pathlib import Path
import json
import traceback

import pandas as pd
import openml
import numpy as np

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch import device
import torch
from autopytorch_utils import data_to_train_test_valid, get_openml_dataset, get_preprocessed_openml_dataset, get_train_test_split

from models.reg_cocktails import get_updates_for_regularization_cocktails

from eval_utils import get_executer

from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes


cocktails = False
try:
    from autoPyTorch.automl_common.common.utils.backend import Backend, create
except ModuleNotFoundError:
    cocktails = True
    from autoPyTorch.utils.backend import Backend, create

MEM_LIMIT = 12000

def transform_key(k):
    """ Transform the key to only have single letters from each word"""
    k = k.replace("data__", "")
    k = "_".join([x[0] for x in k.split("_")])
    return k

def get_incumbent_results(
    run_history_file: str,
    search_space
):
    """
    Get the incumbent configuration and performance from the previous run HPO
    search with AutoPytorch.
    Args:
        run_history_file (str):
            The path where the AutoPyTorch search data is located.
        search_space (ConfigSpace.ConfigurationSpace):
            The ConfigurationSpace that was previously used for the HPO
            search space.
    Returns:
        config, incumbent_run_value (Tuple[ConfigSpace.Configuration, float]):
            The incumbent configuration found from HPO search and the validation
            performance it achieved.
    """
    run_history = get_run_history(run_history_file, search_space)

    run_history_data = run_history.data
    sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
    incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
    config = run_history.ids_config[incumbent_run_key.config_id]
    return config, incumbent_run_value

def get_run_history(run_history_file, search_space):
    from smac.runhistory.runhistory import RunHistory
    run_history = RunHistory()
    run_history.load_json(
        run_history_file,
        search_space,
    )
    return run_history


def get_smac_object(
    scenario_dict,
    seed: int,
    ta,
    ta_kwargs,
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client,
):
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines.
    Args:
        scenario_dict (typing.Dict[str, typing.Any]): constrain on how to run
            the jobs.
        seed (int): to make the job deterministic.
        ta (typing.Callable): the function to be intensified by smac.
        ta_kwargs (typing.Dict[str, typing.Any]): Arguments to the above ta.
        n_jobs (int): Amount of cores to use for this task.
        initial_budget (int):
            The initial budget for a configuration.
        max_budget (int):
            The maximal budget for a configuration.
        dask_client (dask.distributed.Client): User provided scheduler.
    Returns:
        (SMAC4AC): sequential model algorithm configuration object
    """
    from smac.intensification.simple_intensifier import SimpleIntensifier
    from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_ac_facade import SMAC4AC
    # multi-fidelity is disabled, that is why initial_budget and max_budget
    # are not used.
    rh2EPM = RunHistory2EPM4LogCost

    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=None,
        run_id=seed,
        intensifier=SimpleIntensifier,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


def run_on_dataset(args, seed, budget, run_config, max_time):
    run_config = {
        **run_config, 
        'data__dataset_id': args.dataset_id,
    }

    kwargs = {
        k.replace("data__", ""): v for k, v in run_config.items() \
        if k.startswith("data__")
    }

    X, y, categorical_indicator = get_preprocessed_openml_dataset(
        **kwargs
    )
    print(f"Running dataset with config: {run_config}")
    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
         ) = data_to_train_test_valid(X, y, run_config, seed)
    dataset_openml = openml.datasets.get_dataset(args.dataset_id, download_data=False)
    print(f"Running {dataset_openml.name} with train shape: {X_train.shape} and categoricals: {categorical_indicator}")
    exp_dir = args.exp_dir / dataset_openml.name
    search_space_updates, include_updates = get_updates_for_regularization_cocktails(categorical_indicator=categorical_indicator)
    if isinstance(X_train, np.ndarray):
        dtype = "float" if np.issubdtype(X_train.dtype, np.floating) else "int"
        column_types = {}
        for i, is_categorical in enumerate(categorical_indicator):
            if is_categorical:
                column_type = "str"
            else:
                column_type = dtype
            column_types[i] = column_type
        X_train = make_dataframe(X_train, column_types)
        X_valid = make_dataframe(X_valid, column_types)
        X_test = make_dataframe(X_test, column_types)

    api = TabularClassificationTask(
            seed=seed,
            n_jobs=args.nr_workers,
            ensemble_size=1,
            max_models_on_disc=1,
            temporary_directory=exp_dir / "tmp",
            output_directory=exp_dir / "out",
            delete_tmp_folder_after_terminate=False,
            include_components=include_updates,
            search_space_updates=search_space_updates,
            # resampling_strategy_args={'val_share': train_val_split}
        )
    try:
        
        api.set_pipeline_config(# No early stopping and train on gpu
            **{
                'early_stopping': -1,
                'min_epochs': budget,
                'epochs': budget,
                'metrics_during_training': False,
                'device': args.device,
            })
        
        api.search(
                X_train=X_train.copy(),
                y_train=y_train.copy(),
                X_val=X_valid.copy(),
                y_val=y_valid.copy(),
                X_test=X_test.copy(),
                y_test=y_test.copy(),
                dataset_name=dataset_openml.name,
                optimize_metric='accuracy',
                total_walltime_limit=max_time,
                memory_limit=MEM_LIMIT if args.device == "cpu" else None,
                func_eval_time_limit_secs=min(func_eval_time, max_time),
                enable_traditional_pipeline=False,
                get_smac_object_callback=get_smac_object,
                smac_scenario_args={
                    'runcount_limit': args.max_configs,
                },
                all_supported_metrics=False,
                load_models=False
            )


        ############################################################################
        # Refit on the best hp configuration
        # ==================================

        (
        X_train,
        X_test,
        y_train,
        y_test
         ) = get_train_test_split(X, y, max_train_samples=15_000, iter=seed)
        if isinstance(X_train, np.ndarray):
            X_train = make_dataframe(X_train, column_types)
            X_test = make_dataframe(X_test, column_types)
        input_validator = TabularInputValidator(
            is_classification=True,
        )
        input_validator.fit(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
        )

        dataset = TabularDataset(
            X=X_train.copy(),
            Y=y_train.copy(),
            X_test=X_test.copy(),
            Y_test=y_test.copy(),
            dataset_name=dataset_openml.name,
            seed=seed,
            validator=input_validator,
            resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        )
        dataset.is_small_preprocess = True
        print(f"Fitting pipeline with {budget} epochs")

        search_space = api.get_search_space(dataset)
        # only when we perform hpo will there be an incumbent configuration
        # otherwise take a default configuration.

        configuration, incumbent_run_value = get_incumbent_results(
            os.path.join(
                exp_dir,
                'tmp',
                'smac3-output',
                'run_{}'.format(seed),
                'runhistory.json'),
            search_space,
        )
        print(f"Incumbent configuration: {configuration}")
        print(f"Incumbent trajectory: {api.trajectory}")


        fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
            configuration=configuration,
            budget_type='epochs',
            budget=budget,
            dataset=dataset,
            run_time_limit_secs=func_eval_time,
            eval_metric='accuracy',
            memory_limit=MEM_LIMIT if args.device == "cpu" else None,
        )
        torch.cuda.empty_cache()
        X_train = dataset.train_tensors[0]
        y_train = dataset.train_tensors[1]
        X_test = dataset.test_tensors[0]
        y_test = dataset.test_tensors[1]

        test_predictions = fitted_pipeline.predict(X_test, batch_size=1024)

        # test_score = accuracy_score(y_test, test_predictions.squeeze())
        test_score = calculate_score(y_test, test_predictions, metrics=[api._metric], task_type=api.task_type)["accuracy"]

        train_predictions = fitted_pipeline.predict(X_train, batch_size=1024)

        train_score = calculate_score(y_train, train_predictions, metrics=[api._metric], task_type=api.task_type)["accuracy"]

        print(f'Accuracy train: {train_score}, test: {test_score} metric with {run_value}, {run_info}')

        options = vars(args)
        options.pop('exp_dir', None)
        options.pop('autopytorch_source_dir', None)
        final_result = {
            'train_score': train_score,
            'test_score': test_score,
            'dataset_name': dataset_openml.name,
            'dataset_id': args.dataset_id,
            'run_config': run_config,
            'configuration': fitted_pipeline.configuration.get_dictionary(),
            **options
        }
        json.dump(final_result, open(exp_dir / 'result.json', 'w'))
    except Exception as e:
        print(f"Search failed due to {repr(e)}")
        print(traceback.format_exc())
    finally:
        api._cleanup()
    
    return final_result

def make_dataframe(X, column_types):
    X = pd.DataFrame(X, columns=list(column_types.keys()))
    X = X.astype(column_types)
    return X




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
parser.add_argument(
    '--exp_dir',
    type=Path,
    default="/home/rkohli/tabular-benchmark/autopytorch_tmp"
)
parser.add_argument(
    '--max_configs',
    type=int,
    default=10
)
parser.add_argument(
    '--seed',
    type=int,
    default=1
)
parser.add_argument(
    '--epochs',
    type=int,
    default=105
)
parser.add_argument(
    '--nr_workers',
    type=int,
    default=10
)
parser.add_argument(
    '--slurm',
    action='store_true',
    help='True if run parallely on slurm, false otherwise'
)
parser.add_argument(
    "--partition",
    type=str,
    default="bosch_cpu-cascadelake"
)
parser.add_argument(
    '--slurm_job_time_secs',
    type=int,
    default=180,
    help='Time on slurm in seconds')
parser.add_argument(
    '--func_eval_time_secs',
    type=int,
    default=180,
    help='Time on slurm in seconds')
parser.add_argument(
    '--train_samples',
    type=str,
    default="medium",
    help='Number of training samples to use. Can be medium, large',
    choices=['medium', 'large']
)
if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    func_eval_time = args.func_eval_time_secs

    max_time = min(
        (args.max_configs * func_eval_time) / args.nr_workers, 331_200
        )  # max 92 hours

    # CONFIG_DEFAULT = {"train_prop": 0.70,
    #                   "val_test_prop": 0.3,
    #                   "max_val_samples": None,
    #                   "max_test_samples": 50_000,
    #                   "max_train_samples": None,
    #                   "balance": False
    #                   # "max_test_samples": None,
    # }
    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50_000,
                      "max_test_samples": 50_000,
                      "max_train_samples": 10_000 if args.train_samples == "medium" else 50_000,
                      # "max_test_samples": None,
    }
    # preprocessing_config = {
    #     'data__categorical': False,
    #     'data__regression': False,
    #     'data__transformation': None,
    #     'data__remove_pseudo_categorical': True,
    #     'data__remove_high_cardinality_columns': True,
    #     'data__remove_columns_with_nans': True,
    #     'data__balance_classes': True,
    # }
    preprocessing_config = {
        'data__categorical': True,
        'data__regression': False,
        'data__transformation': None,
        'data__remove_pseudo_categorical': False,
        'data__remove_high_cardinality_columns': False,
        'data__remove_columns_with_nans': False,
        'data__balance_classes': False,
    }
    run_config = {**preprocessing_config, **CONFIG_DEFAULT}
    

    
    args.exp_dir = args.exp_dir / "_".join([f"{transform_key(k)}_{v}" for k, v in preprocessing_config.items()])
    job = None
    final_result = None
    if args.slurm:
        slurm_log_folder = args.exp_dir / "log_test"
        slurm_executer = get_executer(
            partition=args.partition,
            log_folder=slurm_log_folder,
            total_job_time_secs=min(int(max_time * 1.5), 345_600),
            cpu_nodes=args.nr_workers,
            gpu=args.device!="cpu",
            mem_per_cpu=MEM_LIMIT)
        job = slurm_executer.submit(run_on_dataset, args, seed, budget, run_config, max_time)
        print(f"Submitted training for {args.dataset_id} with job_id: {job.job_id}")
    else:
        final_result = run_on_dataset(args, seed, budget, run_config, max_time)
    print(f"Waiting for training to finish for {args.dataset_id} with job_id: {job.job_id}")
    try:
        final_result = job.result() if job is not None else final_result
        print(f"Finished training with score: {final_result}")
    except Exception as e:
        print(f"Failed to finish job: {job.job_id} with {repr(e)} and \nConfiguration: {args.dataset_id}")
    
    print(f"Finished search with best score: {final_result}")


