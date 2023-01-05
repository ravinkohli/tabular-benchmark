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

import sklearn.model_selection
from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from generate_dataset_pipeline import generate_dataset
from configs.model_configs.autopytorch_config import autopytorch_config_default, autopytorch_config
from reproduce_utils import get_executer

from autoPyTorch.utils.logging_ import setup_logger, get_named_client_logger, start_log_server

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


def run_random_search(
    args,
    seed,
    budget,
    X_test,
    y_test,
    backend,
    logger_port,
    validator,
    dataset,
    dataset_properties,
    configurations,
    slurm_log_folder
):
    run_history = dict()
    total_job_time = args.slurm_job_time_secs #max(time * 1.5, 120) * args.chunk_size
    if args.slurm:
        slurm_executer = get_executer(
            partition=args.partition,
            log_folder=slurm_log_folder,
            total_job_time_secs=total_job_time,
            gpu=args.device!="cpu",
            mem_per_cpu=15000,
            cpu_nodes=1)
    for i, subset_configurations in enumerate(chunks(configurations, args.nr_workers)):
        current_runs = dict()
        for num_run, config in enumerate(subset_configurations, start=i*args.nr_workers + 1):  # 0 is reserved for refit
            # Run config on dataset
            print(f"Starting training for {num_run} and config: {config}")
            if args.slurm:
                job = slurm_executer.submit(run_on_autopytorch,
                    dataset=copy.copy(dataset),
                    X_test=X_test,
                    y_test=y_test,
                    seed=seed,
                    budget=budget,
                    num_run=num_run,
                    backend=backend,
                    device=args.device,
                    configuration=config,
                    validator=validator,
                    logger_port=logger_port,
                    autopytorch_source_dir=args.autopytorch_source_dir,
                    dataset_properties=dataset_properties,
                    preprocess=args.preprocess
                )
                print(f"Submitted training for {num_run} with job_id: {job.job_id}")
                current_runs[num_run] = {
                    'configuration': config.get_dictionary(),
                    'cost': job
                }
            else:
                final_score = run_on_autopytorch(
                    dataset=copy.copy(dataset),
                    X_test=X_test,
                    y_test=y_test,
                    seed=seed,
                    budget=budget,
                    num_run=num_run,
                    backend=backend,
                    device=args.device,
                    configuration=config,
                    validator=validator,
                    logger_port=logger_port,
                    autopytorch_source_dir=args.autopytorch_source_dir,
                    dataset_properties=dataset_properties,
                    preprocess=args.preprocess
                )
                run_history[num_run] = {
                    'configuration': config.get_dictionary(),
                    'cost': final_score,
                }
                print(f"Finished training with score:{final_score} in {final_score['duration']}")
        if args.slurm:
            for num_run in current_runs:
                if num_run not in run_history:
                    run_history[num_run] = dict()
                run_history[num_run]['configuration'] = current_runs[num_run]['configuration']
                job = current_runs[num_run]['cost']
                print(f"Waiting for training to finish for {num_run} with {job.job_id}")
                try:
                    run_history[num_run]['cost'] = job.result()
                    print(f"Finished training with score: {run_history[num_run]['cost']} in {run_history[num_run]['cost']['duration']}")
                except Exception as e:
                    print(f"Failed to finish job: {job.job_id} with {repr(e)} and \nConfiguration: {current_runs[num_run]['configuration']}")
                    run_history[num_run]['cost'] = {'train': 0, 'test': 0, 'val': 0, 'duration': 0}

    return run_history


def create_logger(
    name: str,
    seed: int,
    temp_dir: str
):
    logger_name = 'AutoPyTorch:%s:%d' % (name, seed)

    # Setup the configuration for the logger
    # This is gonna be honored by the server
    # Which is created below
    setup_logger(
        filename='%s.log' % str(logger_name),
        output_dir=temp_dir,
        logging_config=None
    )

    # As AutoPyTorch works with distributed process,
    # we implement a logger server that can receive tcp
    # pickled messages. They are unpickled and processed locally
    # under the above logging configuration setting
    # We need to specify the logger_name so that received records
    # are treated under the logger_name ROOT logger setting
    context = multiprocessing.get_context('spawn')
    stop_logging_server = context.Event()
    port = context.Value('l')  # be safe by using a long
    port.value = -1

    # "BaseContext" has no attribute "Process" motivates to ignore the attr check
    logging_server = context.Process(  # type: ignore [attr-defined]
        target=start_log_server,
        kwargs=dict(
            host='localhost',
            logname=logger_name,
            event=stop_logging_server,
            port=port,
            filename='%s.log' % str(logger_name),
            output_dir=temp_dir,
            logging_config=None,
        ),
    )

    logging_server.start()

    while True:
        with port.get_lock():
            if port.value == -1:
                time.sleep(0.01)
            else:
                break

    logger_port = int(port.value)

    return logging_server, logger_port, stop_logging_server


def clean_logger(logging_server, stop_logging_server) -> None:
    """
    cleans the logging server created
    Returns:

    """

    # Clean up the logger
    if logging_server.is_alive():
        stop_logging_server.set()

        # We try to join the process, after we sent
        # the terminate event. Then we try a join to
        # nicely join the event. In case something
        # bad happens with nicely trying to kill the
        # process, we execute a terminate to kill the
        # process.
        logging_server.join(timeout=5)
        logging_server.terminate()
        del stop_logging_server


def run_on_dataset(cocktails, args, seed, budget, config):
    config = {
        **config, 
        'data__keyword': args.dataset_id,
    }
    X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(seed))
    dataset_openml = openml.datasets.get_dataset(args.dataset_id, download_data=False)
    print(f"Running {dataset_openml.name} with train shape: {X_train.shape}")
    exp_dir = args.exp_dir / dataset_openml.name
    temp_dir = exp_dir / "tmp_1"
    out_dir = exp_dir /  "out_1"
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

    logging_server, logger_port, stop_logging_server = create_logger(
        name=dataset_openml.name, 
        seed=seed,
        temp_dir=temp_dir)
    logger = get_named_client_logger(name=f"AutoPyTorch:{dataset_openml.name}:{seed}", port=logger_port)
    logger.debug(f"Running {dataset_openml.name} with train shape: {X_train.shape}")
    backend.setup_logger(port=logger_port, name="autopytorch_pipeline")

    validator_kwargs = dict(is_classification=True, logger_port=logger_port)
    if not cocktails:
        validator_kwargs.update({'seed': seed})
    validator = TabularInputValidator(
        **validator_kwargs)
    validator = validator.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_valid,
        y_test=y_valid
    )
    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_valid,
        Y_test=y_valid,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        validator=validator,
        seed=seed,
        dataset_name=dataset_openml.name
    )
    search_space_updates, include_updates = get_updates_for_regularization_cocktails(args.preprocess)
    dataset_requirements = get_dataset_requirements(
                info=dataset.get_required_dataset_info(),
                include=include_updates,
                search_space_updates=search_space_updates
        )
    dataset_properties = dataset.get_dataset_properties(dataset_requirements)

    configuration_space = get_configuration_space(
        info=dataset.get_required_dataset_info(),
        include=include_updates,
        search_space_updates=search_space_updates
    )
    configuration_space.seed(seed)

    configurations = [configuration_space.get_default_configuration()]
    if args.max_configs > 1:
        sampled_configurations = configuration_space.sample_configuration(args.max_configs - 1)
        configurations += sampled_configurations if isinstance(sampled_configurations, list) else [sampled_configurations]

    # number of configurations each worker will evaluate on
    try:
        run_history = run_random_search(
            args,
            seed,
            budget,
            X_test,
            y_test,
            backend,
            logger_port,
            validator,
            dataset,
            dataset_properties,
            configurations=configurations,
            slurm_log_folder=args.exp_dir / "log_test"
        )

        json.dump(run_history, open(temp_dir / 'run_history.json', 'w'))
        sorted_run_history = sorted(run_history, key=lambda x: run_history[x]['cost']['val'], reverse=True)
        print(f"Sorted run history: {sorted_run_history}")
        incumbent_result = run_history[sorted_run_history[0]]
        options = vars(args)
        options.pop('exp_dir', None)
        options.pop('autopytorch_source_dir', None)
        final_result = {
            'test_score': incumbent_result['cost']['test'],
            'train_score': incumbent_result['cost']['train'],
            'val_score': incumbent_result['cost']['val'],
            'dataset_name': dataset_openml.name,
            'dataset_id': args.dataset_id,
            'configuration': incumbent_result['configuration'],
            **options
        }
        json.dump(final_result, open(exp_dir / 'result.json', 'w'))
    except Exception as e:
        print(f"Random Search failed due to {repr(e)}")
        print(traceback.format_exc())
    finally:
        clean_logger(logging_server=logging_server, stop_logging_server=stop_logging_server)
    
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
    '--preprocess',
    action='store_true',
    help='True if we want to run with their preprocessing'
)

if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50_000,
                      "max_test_samples": 50_000,
                      "max_train_samples": 10_000,
                      "balance": True
    }

    config = {
        'data__categorical': False,
        'data__method_name': 'openml',
        'data__impute_nans': True,
        'data__preprocess': args.preprocess}
    
    apt_config = autopytorch_config if args.preprocess else autopytorch_config_default
    config = {**config, **apt_config, **CONFIG_DEFAULT}

    final_result = run_on_dataset(cocktails, args, seed, budget, config)
    print(f"Finished search with best score: {final_result}")

