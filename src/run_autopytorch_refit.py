import argparse
import os
import shutil
import time
import multiprocessing
import copy
from pathlib import Path
import json
import traceback

from ConfigSpace import Configuration
import openml
import numpy as np
import pandas as pd



from models.reg_cocktails import run_on_autopytorch, get_updates_for_regularization_cocktails
from eval_utils import get_executer
from autopytorch_utils import get_data_from_openml_task_id, get_preprocessed_openml_dataset, get_train_test_split
# from run_reproduce import get_preprocessed_train_test_split

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


def run_on_dataset(cocktails, args, seed, budget, dataset_id, config):
    config = {
        **config, 
        'data__keyword': dataset_id,
    }
    kwargs = {
        k: config.get(k, False) \
            if k != "transformation" else config.get(k, None) \
                for k in ["categorical", "regression", "transformation"]}
    # X, y, categorical_indicator = get_preprocessed_openml_dataset(
    #     dataset_id=dataset_id,
    #     **kwargs
    # )
    # dataset_openml = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False)
    # (
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test
    #      ) = get_train_test_split(
    #     X, y,
    #     max_train_samples=config.get("max_train_samples", 15_000),
    #     iter=seed
    # )
    X_train, X_test, y_train, y_test, categorical_indicator, dataset_openml = get_data_from_openml_task_id(task_id=dataset_id)

    print(f"Running {dataset_openml.name} with train shape: {X_train.shape}")
    exp_dir = args.exp_dir / dataset_openml.name
    if (exp_dir / "result.json").exists():
        refit=True
        result_dict = json.load(open(exp_dir / "result.json", "r"))
    else:
        refit=False

    if os.path.exists(exp_dir / "refit_result.json"):
        print("Skipping refit")
        return
    
    temp_dir = exp_dir / "tmp_refit"
    out_dir = exp_dir /  "out_refit"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

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
        X_train=X_train,
        y_train=y_train,
        # X_test=X_valid,
        # y_test=y_valid
    )
    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        # X_test=X_valid,
        # Y_test=y_valid,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        validator=validator,
        seed=seed,
        dataset_name=dataset_openml.name
    )
    dataset.is_small_preprocess = True
    search_space_updates, include_updates = get_updates_for_regularization_cocktails(
        categorical_indicator=categorical_indicator,
    )
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

    if refit:
        configuration = Configuration(configuration_space, result_dict["configuration"])
    else:
        configuration = None
    # number of configurations each worker will evaluate on
    try:
        final_score = run_on_autopytorch(
            dataset=copy.copy(dataset),
            X_test=X_test,
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
            dataset_properties=dataset_properties,
            categorical_indicator=categorical_indicator
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
        final_result = {
            'test_score': final_score['test'],
            'train_score': final_score['train'],
            'val_score': final_score['val'],
            'dataset_name': dataset_openml.name,
            'dataset_id': dataset_id,
            'configuration': configuration.get_dictionary(),
            **options
        }
        json.dump(final_result, open(exp_dir / 'result_refit.json', 'w'))
    finally:
        clean_logger(logging_server=logging_server, stop_logging_server=stop_logging_server)
    
    final_result.pop("configuration", None)
    return final_result


# def get_data_for_refit(seed, dataset_id):
#     openml.datasets.functions._get_dataset_parquet = lambda x: None
#     dataset_openml = openml.datasets.get_dataset(dataset_id=dataset_id)
#     # retrieve categorical data for encoding
#     X, y, categorical_indicator, attribute_names = dataset_openml.get_data(
#         dataset_format="dataframe", target=dataset_openml.default_target_attribute
#     )
#     X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
#     num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
#         preprocessing_clean(X, y, categorical_indicator, categorical=False,
#                         regression=False, transformation=None)
#     train_prop = 0.7
#     train_prop = min(15000 / X.shape[0], train_prop)
#     numeric_transformer = StandardScaler()
#     numeric_transformer_sparse = MaxAbsScaler()
#     categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
#             # ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
#         ]
#     )
#     preprocessor_sparse = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer_sparse,
#                 [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
#             # ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
#         ]
#     )

#     y_train, y_test, _, _, X_train_no_one_hot, X_tes = get_preprocessed_train_test_split(
#                 X,
#                 y,
#                 categorical_indicator,
#                 {'regression': False},
#                 train_prop,
#                 numeric_transformer,
#                 numeric_transformer_sparse,
#                 preprocessor,
#                 preprocessor_sparse,
#                 seed
#             )
            
#     return dataset_openml,y_train,y_test,X_train_no_one_hot,X_test_no_one_hot


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
    '--chunk_size',
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
    
    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "max_test_samples": 50_000,
                      "max_train_samples": None,
                      "balance": True
    }

    config = {
        'data__categorical': False,
        # 'data__impute_nans': True,
        # 'data__preprocess': args.preprocess
    }
    
    config = {**config, **CONFIG_DEFAULT}
    final_benchmark_dataset_ids = [
        42769
        # 44, 151, 293, 
        # 351, 357, 720, 722, 734, 737, 761, 821,
        # 823, 846, 847, 976, 993, 1053, 1120, 1461, 1486, 1489,
        # 1590, 4134, 23517, 41147, 41150, 41162, 42206, 42343, 42395,
        # 42477, 42769, 1044, 1110, 1222, 1476, 1478, 1503, 1526, 4541,
        # 40685, 41168, 41972, 42468
    ]
    #     1222
    #     # 44, 351, 761, 823, 976, 1110, 14, 76, 1478, 1486, 1503, 1526, 23517, 40685
    #     # 151, 293, 722, 821, 993, 1044, 1120, 1461, 1489, 41150, 41168, 42769,
    #     # 60, 354, 357, 720, 734, 735, 737, 816, 819, 833, 846, 847, 979,
    #     # 1053, 1119, 1222, 1242, 1503, 1507, 1590, 4134, 4541, 41147,
    #     # 41162, 41671, 41972, 42206, 42343, 42395, 42468, 42477, 42742, 43489
    # ]

    final_benchmark_dataset_ids = [
        233088, 233090, 233091, 233092, 233093, 233094, 233096, 233099,
        233102, 233103, 233104, 233106, 233107, 233108, 233109, 233110,
        233112, 233113, 233114, 233115, 233116, 233117, 233118, 233119,
        233120, 233121, 233122, 233123, 233124, 233126, 233130, 233131,
        233132, 233133, 233134, 233135, 233137, 233142, 233143, 233146,
    ]
    for i, dataset_ids in enumerate(chunks(final_benchmark_dataset_ids, args.chunk_size)):
        print(f"Running chunk {i}/{np.ceil(len(final_benchmark_dataset_ids) / args.chunk_size)}")
        results = []
        for dataset_id in dataset_ids:
            print(f"Starting refitting on dataset: {dataset_id}")
            if args.slurm:
                results.append(slurm_executer.submit(
                    run_on_dataset, cocktails, args, seed, budget, dataset_id, config))
            else:
                results.append(
                    run_on_dataset(cocktails, args, seed, budget, dataset_id=dataset_id, config=config))
        
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
        try:
            pd.concat(results).to_csv(args.exp_dir / f"refit_results_dirty_{i}.csv")
        except Exception as e:
            print(e)
        # print(f"Refitted with best score: {}")
