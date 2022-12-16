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

from sklearn.model_selection import train_test_split
from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from generate_dataset_pipeline import generate_dataset
from configs.model_configs.autopytorch_config import autopytorch_config_default
from reproduce_utils import get_executer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder
# from run_reproduce import get_preprocessed_train_test_split

from autoPyTorch.utils.logging_ import setup_logger, get_named_client_logger, start_log_server

from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.resampling_strategy import NoResamplingStrategyTypes
from autoPyTorch.utils.pipeline import get_dataset_requirements, get_configuration_space
from preprocessing.preprocessing import preprocessing

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

def get_preprocessed_train_test_split(X, y, categorical_indicator, resnet_config, train_prop, numeric_transformer, numeric_transformer_sparse, preprocessor, preprocessor_sparse, iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop,
                                                                random_state=np.random.RandomState(iter))
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(
                y_train), np.array(y_test)
    if resnet_config["regression"] == True:
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    if X_test.shape[0] > 30000:  # for speed
        indices = np.random.choice(X_test.shape[0], 30000, replace=False)
        try:
            X_test = X_test.iloc[indices]
        except:
            X_test = X_test[indices]
        y_test = y_test[indices]
    try:
        X_train_one_hot = preprocessor.fit_transform(X_train)
        X_test_one_hot = preprocessor.transform(X_test)
        X_train_no_one_hot = np.zeros_like(X_train)
        X_test_no_one_hot = np.zeros_like(X_test)
                # not column transformer to preserve order
        for i in range(X_train.shape[1]):
            if categorical_indicator[i]:
                X_train_no_one_hot[:, i] = X_train[:, i]
                X_test_no_one_hot[:, i] = X_test[:, i]
            else:
                X_train_no_one_hot[:, i] = numeric_transformer.fit_transform(
                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                X_test_no_one_hot[:, i] = numeric_transformer.transform(
                            X_test[:, i].reshape(-1, 1)).reshape(-1)

    except:
        print("trying MaxAbsScaler")
        X_train_one_hot = preprocessor_sparse.fit_transform(X_train)
        X_test_one_hot = preprocessor_sparse.transform(X_test)
        X_train_no_one_hot = np.zeros_like(X_train)
        X_test_no_one_hot = np.zeros_like(X_test)
        for i in range(X_train.shape[1]):
            if categorical_indicator[i]:
                X_train_no_one_hot[:, i] = X_train[:, i]
                X_test_no_one_hot[:, i] = X_test[:, i]
            else:
                X_train_no_one_hot[:, i] = numeric_transformer_sparse.fit_transform(
                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                X_test_no_one_hot[:, i] = numeric_transformer_sparse.transform(
                            X_test[:, i].reshape(-1, 1)).reshape(-1)

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    X_train_one_hot, X_test_one_hot = np.array(X_train_one_hot), np.array(X_test_one_hot)
    X_train_no_one_hot, X_test_no_one_hot = np.array(X_train_no_one_hot), np.array(
                X_test_no_one_hot)
    X_train_one_hot, X_test_one_hot = X_train_one_hot.astype(np.float32), X_test_one_hot.astype(
                np.float32)
    X_train_no_one_hot, X_test_no_one_hot = X_train_no_one_hot.astype(
                np.float32), X_test_no_one_hot.astype(np.float32)
        
    return y_train,y_test,X_train_one_hot,X_test_one_hot,X_train_no_one_hot,X_test_no_one_hot


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
            gpu=args.device!="cpu")
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
                    dataset_properties=dataset_properties
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
                    dataset_properties=dataset_properties
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


def run_on_dataset(cocktails, args, seed, budget, dataset_id):
    dataset_openml = openml.datasets.get_dataset(dataset_id=dataset_id)
    # retrieve categorical data for encoding
    X, y, categorical_indicator, attribute_names = dataset_openml.get_data(
        dataset_format="dataframe", target=dataset_openml.default_target_attribute
    )
    X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
    num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
        preprocessing(X, y, categorical_indicator, categorical=False,
                        regression=False, transformation=None)
    train_prop = 0.7
    train_prop = min(15000 / X.shape[0], train_prop)
    numeric_transformer = StandardScaler()
    numeric_transformer_sparse = MaxAbsScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
        ]
    )
    preprocessor_sparse = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_sparse,
                [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
        ]
    )

    y_train, y_test, _, _, X_train_no_one_hot, X_test_no_one_hot = get_preprocessed_train_test_split(
                X,
                y,
                categorical_indicator,
                {'regression': False},
                train_prop,
                numeric_transformer,
                numeric_transformer_sparse,
                preprocessor,
                preprocessor_sparse,
                1
            )
    # config = {
    #     **config, 
    #     'data__keyword': args.dataset_id,
    # }
    # X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(seed))
    print(f"Running {dataset_openml.name} with train shape: {X_train_no_one_hot.shape}")
    exp_dir = args.exp_dir / dataset_openml.name
    result_dict = json.load(open(exp_dir / "result.json", "r"))

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

    configuration_space = get_configuration_space(
        info=dataset.get_required_dataset_info(),
        include=include_updates,
        search_space_updates=search_space_updates
    )
    configuration_space.seed(seed)

    configuration = Configuration(configuration_space, result_dict["configuration"])
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
            'val_score': final_score['val'],
            'dataset_name': dataset_openml.name,
            'dataset_id': dataset_id,
            'configuration': configuration,
            **options
        }
        json.dump(final_result, open(exp_dir / 'result_refit.json', 'w'))
    except Exception as e:
        print(f"Refit failed due to {repr(e)}")
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


if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    budget = args.epochs

    
    final_benchmark_dataset_ids = [44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131, 44089, 44090, 44091]
    for dataset_id in final_benchmark_dataset_ids:
        final_result = run_on_dataset(cocktails, args, seed, budget, dataset_id=dataset_id)
        print(f"Refitted with best score: {final_result}")

