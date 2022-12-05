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

import sklearn.model_selection
from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails

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
    start=1
):
    run_history = dict()
    for num_run, config in enumerate(configurations, start=start):  # 0 is reserved for refit
        # Run config on dataset
        start_time = time.time()
        print(f"Starting training for {num_run} and config: {config}")
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
            dataset_properties=dataset_properties
        )
        duration = time.time()-start_time
        run_history[num_run] = {
            'configuration': config.get_dictionary(),
            'score': final_score,
            'time': duration
        }
        print(f"Finished training with score:{final_score} in {duration}")
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

if __name__ == '__main__':
    args = parser.parse_args()
    SEED = 1
    BUDGET = 5
    dataset_openml = openml.datasets.get_dataset(args.dataset_id)
    X, y, categorical_indicator, attribute_names = dataset_openml.get_data(
        target=dataset_openml.default_target_attribute
    )
    X_train_original, X_test, y_train_original, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=SEED,
    )
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X_train_original,
        y_train_original,
        random_state=SEED,
    )

    out_dir = args.exp_dir / dataset_openml.name
    temp_dir = out_dir / "tmp_1"
    out_dir = out_dir /  "out_1"
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
        seed=SEED,
        temp_dir=temp_dir)
    backend.setup_logger(port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, name="autopytorch_pipeline")

    validator_kwargs = dict(is_classification=True, logger_port=logger_port)
    if not cocktails:
        validator_kwargs.update({'seed': SEED})
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
        seed=SEED,
        dataset_name=dataset_openml.name
    )
    search_space_updates, include_updates = get_updates_for_regularization_cocktails()
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
    configuration_space.seed(SEED)

    configurations = [configuration_space.sample_configuration() for i in range(args.max_configs)]

    start_time = time.time()
    try:
        run_history = run_random_search(
            args,
            SEED,
            BUDGET,
            X_test,
            y_test,
            backend,
            logger_port,
            validator,
            dataset,
            dataset_properties,
            configurations
        )
        
        json.dump(run_history, open(out_dir / 'run_history.json', 'w'))
    except Exception as e:
        print(f"Random Search failed due to {repr(e)}")
        print(traceback.format_exc())

    sorted_run_history = sorted(run_history, key=lambda x: run_history[x]['score']['val'], reverse=True)
    print(f"Sorted run history: {sorted_run_history}")
    incumbent_configuration = run_history[sorted_run_history[0]]['configuration']

    print(f"Refitting with : {incumbent_configuration}")
    validator_kwargs = dict(is_classification=True, logger_port=logger_port)
    if not cocktails:
        validator_kwargs.update({'seed': SEED})
    validator = TabularInputValidator(
        **validator_kwargs)
    validator = validator.fit(
        X_train=X_train_original,
        y_train=y_train_original,
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
        seed=SEED,
        dataset_name=dataset_openml.name
    )

    final_score = run_on_autopytorch(
        dataset=copy.copy(dataset),
        X_test=None,
        y_test=None,
        seed=SEED,
        budget=BUDGET,
        num_run=0,
        backend=backend,
        device=args.device,
        configuration=Configuration(configuration_space=configuration_space, values=incumbent_configuration),
        validator=validator,
        logger_port=logger_port,
        autopytorch_source_dir=args.autopytorch_source_dir,
        dataset_properties=dataset_properties
    )

    print(f"Finished refitting with score:{final_score} in {time.time() - start_time}")

    clean_logger(logging_server=logging_server, stop_logging_server=stop_logging_server)
