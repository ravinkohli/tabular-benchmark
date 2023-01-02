import argparse

import copy
from pathlib import Path
import json
import traceback

from ConfigSpace import ConfigurationSpace
import numpy as np
import pandas as pd

from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.utils.pipeline import get_dataset_requirements, get_configuration_space

from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from reproduce_utils import get_executer
from run_autopytorch_refit import get_data_for_refit, create_logger, clean_logger

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

def run_on_dataset(cocktails, args, seed, budget, dataset_id):
    dataset_openml, y_train, y_test, X_train_no_one_hot, X_test_no_one_hot = get_data_for_refit(seed, dataset_id)
    # config = {
    #     **config, 
    #     'data__keyword': args.dataset_id,
    # }
    # X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(seed))
    print(f"Running {dataset_openml.name} with train shape: {X_train_no_one_hot.shape}")
    exp_dir = args.exp_dir / dataset_openml.name / str(dataset_id)

    temp_dir = exp_dir / "tmp_refit"
    out_dir = exp_dir /  "out_refit"
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
    backend.setup_logger(port=logger_port, name="autopytorch_pipeline")

    validator_kwargs = dict(is_classification=True, logger_port=logger_port)
    if not cocktails:
        validator_kwargs.update({'seed': seed})
    validator = TabularInputValidator(
        **validator_kwargs)
    validator = validator.fit(
        X_train=X_train_no_one_hot,
        y_train=y_train,
        # X_test=X_valid,
        # y_test=y_valid
    )
    dataset = TabularDataset(
        X=X_train_no_one_hot,
        Y=y_train,
        # X_test=X_valid,
        # Y_test=y_valid,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        validator=validator,
        seed=seed,
        dataset_name=dataset_openml.name
    )
    search_space_updates, include_updates = get_updates_for_regularization_cocktails()
    dataset_requirements = get_dataset_requirements(
                info=dataset.get_required_dataset_info(),
                include=include_updates,
                search_space_updates=search_space_updates
        )
    dataset_properties = dataset.get_dataset_properties(dataset_requirements)

    configuration_space: ConfigurationSpace = get_configuration_space(
        info=dataset.get_required_dataset_info(),
        include=include_updates,
        search_space_updates=search_space_updates
    )
    configuration_space.seed(seed)
    configuration = configuration_space.get_default_configuration()
    # number of configurations each worker will evaluate on
    try:
        final_score = run_on_autopytorch(
            dataset=copy.copy(dataset),
            X_test=X_test_no_one_hot,
            y_test=y_test,
            seed=seed,
            budget=budget,
            num_run=0,
            backend=backend,
            device=args.device,
            configuration=configuration,
            validator=validator,
            logger_port=logger_port,
            autopytorch_source_dir=args.autopytorch_source_dir,
            dataset_properties=dataset_properties
        )


        options = vars(args)
        options.pop('exp_dir', None)
        options.pop('autopytorch_source_dir', None)
        final_result = {
            'test_score': final_score['test'],
            'train_score': final_score['train'],
            'dataset_name': dataset_openml.name,
            'dataset_id': dataset_id,
            'configuration': configuration,
            **options
        }
        json.dump(final_result, open(exp_dir / 'result_refit.json', 'w'))
    except Exception as e:
        final_result = {
            'test_score': final_score['test'],
            'train_score': final_score['train'],
            'dataset_name': dataset_openml.name,
            'dataset_id': dataset_id,
            'configuration': configuration.get_dictionary(),
            **options
        }
        json.dump(final_result, open(exp_dir / 'result_refit.json', 'w'))
        print(f"Refit failed due to {repr(e)}")
        print(traceback.format_exc())
    finally:
        clean_logger(logging_server=logging_server, stop_logging_server=stop_logging_server)
    
    final_result.pop("configuration", None)
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


if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    total_job_time = args.slurm_job_time_secs #max(time * 1.5, 120) * args.chunk_size
    if args.slurm:
        slurm_executer = get_executer(
            partition=args.partition,
            log_folder=args.exp_dir / "log_test",
            total_job_time_secs=total_job_time,
            gpu=args.device!="cpu")
    final_benchmark_dataset_ids = [
               44,    60,   151,   279,   293,   351,   354,   357,   720,
              722,   725,   734,   735,   737,   761,   803,   816,   819,
              821,   823,   833,   846,   847,   871,   976,   979,   993,
             1044,  1053,  1110,  1113,  1119,  1120,  1222,  1241,  1242,
             1461,  1476,  1477,  1478,  1486,  1489,  1503,  1507,  1526,
             1590,  4134,  4541, 23517, 40685, 40923, 41146, 41147, 41150,
            41162, 41163, 41164, 41166, 41168, 41169, 41671, 41972, 42206,
            42343, 42395, 42468, 42477, 42742, 42746, 42769, 43489, 44089,
            44090, 44091]

    results = []
    for dataset_id in final_benchmark_dataset_ids:
        print(f"Starting refitting on dataset: {dataset_id}")
        if args.slurm:
            results.append(slurm_executer.submit(run_on_dataset, cocktails, args, seed, budget, dataset_id))
        else:
            results.append(run_on_dataset(cocktails, args, seed, budget, dataset_id=dataset_id))
    
    if args.slurm:
        new_results = []
        for job in results:
            print(f"Waiting for job to finish: {job.job_id}")
            try:
                result = job.result()
                new_results.append(result)
            except Exception as e:
                print(f"Failed for {job.job_id} with exception: {repr(e)}")
                print(traceback.format_exc())

        results = new_results
    
    pd.concat(results).to_csv(args.exp_dir / "refit_results_3.csv")
        # print(f"Refitted with best score: {}")

