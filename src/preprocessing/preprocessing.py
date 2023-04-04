from sklearn.preprocessing import LabelEncoder
from preprocessing.utils import *
import numpy as np


def impute_nans(X, categorical_indicator):
    from sklearn.impute import SimpleImputer
    # Impute numerical columns with mean and categorical columns with most frequent
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    numerical_imputer = SimpleImputer(strategy="mean")
    # check that there a > 0 categorical columns
    if sum(categorical_indicator) > 0:
        X.iloc[:, categorical_indicator] = categorical_imputer.fit_transform(X.iloc[:, categorical_indicator])
    # check that there a > 0 numerical columns
    if sum(~categorical_indicator) > 0:
        X.iloc[:, ~categorical_indicator] = numerical_imputer.fit_transform(X.iloc[:, ~categorical_indicator])
    return X


def preprocessing(
        X,
        y,
        categorical_indicator,
        categorical,
        regression,
        remove_pseudo_categorical=True,
        remove_high_cardinality_columns=True,
        remove_columns_with_nans=True,
        balance_classes=True,
        categorify_binary=True,
        transformation=None,
        dataset_id=None,
        transform_y_back=False
    ):
    print("dataset_id: ", dataset_id)
    original_n_samples, original_n_features = X.shape
    le = LabelEncoder()
    if not regression:
        y = le.fit_transform(y)
    if categorify_binary:
        binary_variables_mask = np.array(X.nunique() == 2)
        for i in range(X.shape[1]):
            if binary_variables_mask[i]:
                categorical_indicator[i] = True

    unwanted_columns = find_unwanted_columns(X, dataset_id) #TODO remove this
    print("unwanted_columns: ", unwanted_columns)
    specific_categorical = specify_categorical(X, dataset_id)
    print("specific_categorical: ", specific_categorical)
    for i in range(X.shape[1]):
        if categorical_indicator[i]:
            continue
        for index in range(X.shape[0]):
            if not pd.isnull(X.iloc[index, i]):
                if type(X.iloc[index, i]) == str:
                    categorical_indicator[i] = True
                break
        else:
            print("Column {} is empty".format(X.columns[i]))

    if not dataset_id is None:
        # Returns the list of specific categorical variables not indicated as categorical
        specific_categorical = specify_categorical(X, dataset_id)
        for i in range(X.shape[1]):
            if X.columns[i] in specific_categorical:
                categorical_indicator[i] = True
    
    cols_to_delete = []
    
    n_pseudo_categorical = 0
    if remove_pseudo_categorical:
        pseudo_categorical_mask = np.array(X.nunique() < 10)
        for i in range(X.shape[1]):
            if pseudo_categorical_mask[i]:
                if not categorical_indicator[i]:
                    n_pseudo_categorical += 1
                    cols_to_delete.append(i)
        print("Number of pseudo categorical variables: {}".format(n_pseudo_categorical))

    if not categorical:
        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                cols_to_delete.append(i)

    if not dataset_id is None:
        unwanted_columns = find_unwanted_columns(X, dataset_id)
        print("Number of unwanted columns: {}".format(len(unwanted_columns)))
        for i in range(X.shape[1]):
            if X.columns[i] in unwanted_columns:
                cols_to_delete.append(i)
    if len(cols_to_delete) > 0:
        print("cols to delete")
        print(X.columns[cols_to_delete])
        print("{} columns removed".format(len(cols_to_delete)))
        X = X.drop(X.columns[cols_to_delete], axis=1)
        categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                                not i in cols_to_delete]  # update categorical indicator

    num_high_cardinality = 0
    if remove_high_cardinality_columns:
        X, y, categorical_indicator, num_high_cardinality = remove_high_cardinality(X, y, categorical_indicator, 20)


    num_columns_missing = 0
    num_rows_missing = 0
    if remove_columns_with_nans:
        X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y)
        categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                                not missing_cols_mask[i]]
    else:
        X = impute_nans(X, np.array(categorical_indicator))

    if X.shape[0] > 1:
        if not regression:
            if balance_classes:
                X, y = balance(X, y)
                # assert len(X) == len(y)
                assert len(np.unique(y)) == 2
                assert np.max(y) == 1
            y = le.fit_transform(y)

        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                X.iloc[:, i] = LabelEncoder().fit_transform(X.iloc[:, i])

        if transformation is not None and transformation != "none":
            assert regression
            y = transform_target(y, transformation)
        else:
            print("NO TRANSFORMATION")

    if transform_y_back and not regression:
        y = le.inverse_transform(y)

    num_categorical_columns = sum(categorical_indicator)
    print("Number of categorical columns: {}".format(num_categorical_columns))

    return X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, num_categorical_columns, \
           n_pseudo_categorical, original_n_samples, original_n_features
