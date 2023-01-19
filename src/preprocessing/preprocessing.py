from sklearn.preprocessing import LabelEncoder
from preprocessing.utils import *

def preprocessing(X, y, categorical_indicator, categorical, regression, transformation=None):
    num_categorical_columns = sum(categorical_indicator)
    original_n_samples, original_n_features = X.shape
    le = LabelEncoder()
    if not regression:
        y = le.fit_transform(y)
    binary_variables_mask = X.nunique() == 2
    for i in range(X.shape[1]):
        if binary_variables_mask[i]:
            categorical_indicator[i] = True
    for i in range(X.shape[1]):
        if type(X.iloc[1, i]) == str:
            categorical_indicator[i] = True

    pseudo_categorical_mask = X.nunique() < 10
    n_pseudo_categorical = 0
    cols_to_delete = []
    # for i in range(X.shape[1]):
    #     if pseudo_categorical_mask[i]:
    #         if not categorical_indicator[i]:
    #             n_pseudo_categorical += 1
    #             cols_to_delete.append(i)
    if not categorical:
        for i in range(X.shape[1]):
            if categorical_indicator[i]:
                cols_to_delete.append(i)
    print("low card to delete")
    print(X.columns[cols_to_delete])
    print("{} low cardinality numerical removed".format(n_pseudo_categorical))
    X = X.drop(X.columns[cols_to_delete], axis=1)
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                             not i in cols_to_delete] # update categorical indicator
    print("Remaining categorical")
    print(categorical_indicator)
    num_high_cardinality = 0
    # X, y, categorical_indicator, num_high_cardinality = remove_high_cardinality(X, y, categorical_indicator, 20)
    # print([X.columns[i] for i in range(X.shape[1]) if categorical_indicator[i]])
    num_columns_missing, num_rows_missing = (None, None)
    X = impute_missing_values(X) #, y, 0.2)
    X = X.toarray() if hasattr(X, 'toarray') else X
    X = pd.DataFrame(X) 
    # X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y, 0.2)
    # categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
    #                          not missing_cols_mask[i]]
    if not regression:
        X, y = balance(X, y)
        y = le.fit_transform(y)
        assert len(X) == len(y)
        assert len(np.unique(y)) == 2
        assert np.max(y) == 1
    for i in range(X.shape[1]):
        if categorical_indicator[i]:
            X.iloc[:, i] = LabelEncoder().fit_transform(X.iloc[:, i])

    if transformation is not None:
        assert regression
        y = transform_target(y, transformation)
    else:
        print("NO TRANSFORMATION")

    return X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, num_categorical_columns, \
              n_pseudo_categorical, original_n_samples, original_n_features
    # return X, y, categorical_indicator, None, None, None, num_categorical_columns, \
    #           None, original_n_samples, original_n_features
    