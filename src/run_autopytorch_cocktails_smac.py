import argparse
import time
import logging.handlers
import multiprocessing
import copy
import os
from pathlib import Path
import json
import traceback

from ConfigSpace import Configuration
import openml
import numpy as np

from sklearn.metrics import accuracy_score


from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from generate_dataset_pipeline import generate_dataset, generate_dataset_smac
from configs.model_configs.autopytorch_config import autopytorch_config_default
from reproduce_utils import get_executer
from run_autopytorch_refit import get_data_for_refit

from autoPyTorch.utils.logging_ import setup_logger, get_named_client_logger, start_log_server
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.utils.pipeline import get_dataset_requirements, get_configuration_space

cocktails = False
try:
    from autoPyTorch.automl_common.common.utils.backend import Backend, create
except ModuleNotFoundError:
    cocktails = True
    from autoPyTorch.utils.backend import Backend, create


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
    from smac.runhistory.runhistory import RunHistory
    run_history = RunHistory()
    run_history.load_json(
        run_history_file,
        search_space,
    )

    run_history_data = run_history.data
    sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
    incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
    config = run_history.ids_config[incumbent_run_key.config_id]
    return config, incumbent_run_value


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



def run_on_dataset(args, seed, budget, config, max_time):
    config = {
        **config, 
        'data__keyword': args.dataset_id,
    }

    X_train, X_test, y_train, y_test, categorical_indicator, train_val_split = generate_dataset_smac(config, np.random.RandomState(seed))

    dataset_openml = openml.datasets.get_dataset(args.dataset_id, download_data=False)
    print(f"Running {dataset_openml.name} with train shape: {X_train.shape}")
    exp_dir = args.exp_dir / dataset_openml.name
    search_space_updates, include_updates = get_updates_for_regularization_cocktails()

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
            resampling_strategy_args={'val_share': train_val_split}
        )
    try:
        
        api.set_pipeline_config(# No early stopping and train on gpu
            **{
                'early_stopping': -1,
                'min_epochs': budget,
                'epochs': budget,
            })
        
        api.search(
                X_train=X_train.copy(),
                y_train=y_train.copy(),
                X_test=X_test.copy(),
                y_test=y_test.copy(),
                dataset_name=dataset_openml.name,
                optimize_metric='accuracy',
                total_walltime_limit=max_time,
                memory_limit=12000,
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
        dataset_openml, y_train, y_test, X_train, X_test = get_data_for_refit(seed, args.dataset_id)

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
        dataset.is_small_preprocess = False
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
            memory_limit=12000,
        )

        X_train = dataset.train_tensors[0]
        y_train = dataset.train_tensors[1]
        X_test = dataset.test_tensors[0]
        y_test = dataset.test_tensors[1]

        test_predictions = fitted_pipeline.predict(X_test)

        # test_score = accuracy_score(y_test, test_predictions.squeeze())
        test_score = calculate_score(y_test, test_predictions, metrics=[api._metric], task_type=api.task_type)["accuracy"]

        train_predictions = fitted_pipeline.predict(X_train)

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
            'configuration': fitted_pipeline.configuration.get_dictionary(),
            **options
        }
        json.dump(final_result, open(exp_dir / 'result.json', 'w'))
    except Exception as e:
        print(f"Random Search failed due to {repr(e)}")
        print(traceback.format_exc())
    finally:
        api._cleanup()
    
    return final_result


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


if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    func_eval_time = args.func_eval_time_secs

    max_time = (args.max_configs * func_eval_time) / args.nr_workers

    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50_000,
                      "max_test_samples": 50_000,
                      "max_train_samples": 10_000,
                      "balance": True
                      # "max_test_samples": None,
    }
    config = {
        'data__categorical': False,
        'data__method_name': 'openml',
        'data__impute_nans': True}
    config = {**config, **autopytorch_config_default, **CONFIG_DEFAULT}
    
    job = None
    final_result = None
    if args.slurm:
        slurm_log_folder = args.exp_dir / "log_test"
        slurm_executer = get_executer(
            partition=args.partition,
            log_folder=slurm_log_folder,
            total_job_time_secs=int(max_time * 1.5),
            cpu_nodes=args.nr_workers,
            gpu=args.device!="cpu")
        job = slurm_executer.submit(run_on_dataset, args, seed, budget, config, max_time)
        print(f"Submitted training for {args.dataset_id} with job_id: {job.job_id}")
    else:
        final_result = run_on_dataset(args, seed, budget, config, max_time)
    print(f"Waiting for training to finish for {args.dataset_id} with job_id: {job.job_id}")
    try:
        final_result = job.result() if job is not None else final_result
        print(f"Finished training with score: {final_result}")
    except Exception as e:
        print(f"Failed to finish job: {job.job_id} with {repr(e)} and \nConfiguration: {args.dataset_id}")
    
    print(f"Finished search with best score: {final_result}")


