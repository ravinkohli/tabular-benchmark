from __future__ import annotations

import argparse
import time
import multiprocessing
import copy
import os
from pathlib import Path
import json
import traceback

from ConfigSpace import Configuration, ConfigurationSpace
import openml
import numpy as np
import pandas as pd

from typing import Tuple

from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from generate_dataset_pipeline import generate_dataset
from configs.model_configs.autopytorch_config import autopytorch_config
from reproduce_utils import get_executer
from utils.dataset_utils import Dataset
from autoPyTorch.utils.logging_ import setup_logger, get_named_client_logger, start_log_server

from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.utils.pipeline import get_dataset_requirements, get_configuration_space
from autoPyTorch.utils.backend import Backend, create


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class BaseDatasetRunner(object):
    def __init__(
        self,
        dataset_id: int,
        seed: int,
        budget: int,
        exp_dir: Path,
        autopytorch_source_dir: str,
        device: str = 'cpu',
        nr_workers: None | int = None,
        max_configs: int = 20,
    ) -> None:

        self.dataset_name: str =  openml.datasets.get_dataset(dataset_id, download_data=False).name
        self.dataset_id = dataset_id
        self.seed = seed
        self.budget = budget
        self.exp_dir = exp_dir
        self.nr_workers = nr_workers
        self.max_configs = max_configs
        self.autopytorch_source_dir = autopytorch_source_dir
        self.device = device

        self.create_logger()
        self.backend = self._create_backend()

    def create_logger(
        self,
        temp_dir: str
    ) -> None:
        logger_name = 'AutoPyTorch:%s:%d' % (self.dataset.name, self.seed)

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
        self.stop_logging_server_ = context.Event()
        port = context.Value('l')  # be safe by using a long
        port.value = -1

        # "BaseContext" has no attribute "Process" motivates to ignore the attr check
        self.logging_server_ = context.Process(  # type: ignore [attr-defined]
            target=start_log_server,
            kwargs=dict(
                host='localhost',
                logname=logger_name,
                event=self.stop_logging_server_,
                port=port,
                filename='%s.log' % str(logger_name),
                output_dir=temp_dir,
                logging_config=None,
            ),
        )

        self.logging_server_.start()

        while True:
            with port.get_lock():
                if port.value == -1:
                    time.sleep(0.01)
                else:
                    break

        self.logger_port_ = int(port.value)

    def clean_logger(
        self
    ) -> None:
        """
        cleans the logging server created
        Returns:

        """

        # Clean up the logger
        if self.logging_server_.is_alive():
            self.stop_logging_server_.set()

            # We try to join the process, after we sent
            # the terminate event. Then we try a join to
            # nicely join the event. In case something
            # bad happens with nicely trying to kill the
            # process, we execute a terminate to kill the
            # process.
            self.logging_server_.join(timeout=5)
            self.logging_server_.terminate()
            del self.stop_logging_server_

    def _create_backend(self) -> Backend:
        temp_dir = self.exp_dir / f"tmp_{time.time()}"
        out_dir = self.exp_dir /  f"out_{time.time()}"
        backend_kwargs = dict(
            temporary_directory=temp_dir,
            output_directory=out_dir,
            delete_output_folder_after_terminate=False,
            delete_tmp_folder_after_terminate=False
        )
        backend = create(
            **backend_kwargs
        )
        backend.setup_logger(port=self.logger_port_, name="autopytorch_pipeline")
        return backend

    def _setup_experiment(
        self,
        config: dict,
    ) -> Tuple[list[Configuration], Dataset, TabularDataset]:

        dataset = Dataset.fetch(dataset_id=self.dataset_id, seed=self.seed, dataset_name=self.dataset_name, **config)

        validator_kwargs = dict(is_classification=True, logger_port=self.logger_port_)

        self.validator_ = TabularInputValidator(
            **validator_kwargs)
        self.validator_ = self.validator_.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_valid,
            y_test=dataset.y_valid
        )
        autopytorch_datamnager = TabularDataset(
            X=dataset.X_train,
            Y=dataset.y_train,
            X_test=dataset.X_valid,
            Y_test=dataset.y_valid,
            resampling_strategy=NoResamplingStrategyTypes.no_resampling,
            validator=self.validator_,
            seed=self.seed,
            dataset_name=dataset.name
        )
        self.search_space_updates_, self.include_updates_ = get_updates_for_regularization_cocktails()
        dataset_requirements = get_dataset_requirements(
                    info=autopytorch_datamnager.get_required_dataset_info(),
                    include=self.include_updates_,
                    search_space_updates=self.search_space_updates_
            )
        self.dataset_properties_ = autopytorch_datamnager.get_dataset_properties(dataset_requirements)

        self.configuration_space_ = get_configuration_space(
            info=autopytorch_datamnager.get_required_dataset_info(),
            include=self.include_updates_,
            search_space_updates=self.search_space_updates_
        )
        self.configuration_space_.seed(self.seed)
        return (
            self.configuration_space_.sample_configuration(self.max_configs),
            dataset,
            autopytorch_datamnager
        )

    def run_experiment(
        self,
        config: dict
    ):
        configurations, dataset, autopytorch_dataset = self._setup_experiment(config)
        run_history = dict()
        for num_run, configuration in enumerate(configurations, start=1):  # 0 is reserved for refit
            # Run config on dataset
            start_time = time.time()
            print(f"Starting training for {num_run} and config: {config}")
            final_score = run_on_autopytorch(
                dataset=copy.copy(autopytorch_dataset),
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                seed=seed,
                budget=budget,
                num_run=num_run,
                backend=self.backend,
                device=self.device,
                configuration=configuration,
                validator=self.validator_,
                logger_port=self.logger_port_,
                autopytorch_source_dir=args.autopytorch_source_dir,
                dataset_properties=self.dataset_properties_
            )
            duration = time.time()-start_time
            run_history[num_run] = {
                'configuration': config.get_dictionary(),
                'score': final_score,
                'time': duration
            }
            print(f"Finished training with score:{final_score} in {duration}")



def run_on_dataset(args, seed, budget, config):
    X_test, y_test, dataset_openml, exp_dir, temp_dir, backend, logging_server, logger_port, stop_logging_server, validator, dataset, dataset_properties, configurations = setup_experiment(args, seed, config=config)

    log_folder = os.path.join(args.exp_dir, "log_test/")

    if args.slurm:
        total_job_time = args.slurm_job_time #max(time * 1.5, 120) * args.chunk_size
        slurm_executer = get_executer(
            partition=args.partition,
            log_folder=log_folder,
            total_job_time_secs=total_job_time,
            gpu=args.device!="cpu")

    # number of configurations each worker will evaluate on
    # n_config_per_chunk = np.ceil(args.max_configs / args.nr_workers)
    run_history = dict()
    for i, subset_configurations in enumerate(chunks(configurations, args.nr_workers)):
        try:
            subset_run_history = slurm_executer.submit(run_random_search,
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
                subset_configurations,
                start=i*args.nr_workers + 1
            )
            
            run_history = {**run_history, **subset_run_history}

            json.dump(run_history, open(temp_dir / 'run_history.json', 'w'))
        except Exception as e:
            print(f"Random Search failed due to {repr(e)}")
            print(traceback.format_exc())

    sorted_run_history = sorted(run_history, key=lambda x: run_history[x]['score']['val'], reverse=True)
    print(f"Sorted run history: {sorted_run_history}")
    incumbent_result = run_history[sorted_run_history[0]]

    clean_logger(logging_server=logging_server, stop_logging_server=stop_logging_server)
    options = vars(args)
    options.pop('exp_dir', None)
    options.pop('autopytorch_source_dir', None)
    final_result = {
        'test_score': incumbent_result['score']['test'],
        'train_score': incumbent_result['score']['train'],
        'dataset_name': dataset_openml.name,
        'dataset_id': args.dataset_id,
        'configuration': incumbent_result['configuration'],
        **options
    }
    json.dump(final_result, open(exp_dir / 'result.json', 'w'))
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
    '--slurm_job_time',
    type=int,
    default=60,
    help='Time on slurm in minutes')

# parser.add_argument(
#     '--chunk_size',
#     type=int,
#     default=4
# )


if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50000,
                      "max_test_samples": 50000,
                      "max_train_samples": None,
                      # "max_test_samples": None,
    }
    config = {
        'data__categorical': False,
        'data__method_name': 'openml',
        'data__impute_nans': True}
    config = {**config, **autopytorch_config, **CONFIG_DEFAULT}
    
    
    final_result = run_on_dataset(args, seed, budget, config)
    print(f"Finished search with best score: {final_result}")

