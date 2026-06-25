import os
import pdb
import math
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.impute import SimpleImputer
import openml
#openml.config.server = "http://145.38.195.79/api/v1/xml" # Point to the read-only server
from loguru import logger

# TODO
# organize teh dataset_config for the load_data API.
# dataset_config = {
# 'dataname': { 'cat':[],'bin':[], 'num':[], 
# 'cols':[]}
# }


OPENML_DATACONFIG = {
    # 'credit-g': {'bin': ['own_telephone', 'foreign_worker']},
    # 'credit-approval': {'bin': ['A1', 'A9', 'A10', 'A12']},
    # 'cylinder-bands' : {'bin' : ['Date', 
    #                              'Relevant Information', 
    #                              'responsibilities require the expert to do what an expert does best',
    #                              'Number of Instances',
    #                              'Number of Attributes',
    #                              'Attribute Information',
    #                              'timestamp']},
    # 'heart-statlog' : {'bin' : ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']},
    # 'bank-marketing' : {'bin' : ['default', 'housing', 'loan']},
    # 'ilpd' : {'bin' : ['V2']},
    # 'hepatits' : {'bin' : ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 
    #     'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 
    #     'SPLEEN_PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']},
    # 'stroke' : {'bin' : ['gender', 'ever_married', 'Residence_type', 'hypertension', 'heart_disease']},
    # 'Cardiovascular-Disease-dataset' : {'bin' : ['gender', 'active', 'alco', 'smoke']},
    # 'adult': {'bin' : ['sex']},
    # 'sick': {'bin': ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
    #               'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
    #               'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
    #               'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 
    #               'TBG_measured']},
    # 'cmc': {'bin': ['Wifes_religion', 'Wifes_now_working%3F', 'Media_exposure']}
    
    
    
}

EXAMPLE_DATACONFIG = {
    "example": {
        "bin": ["bin1", "bin2"],
        "cat": ["cat1", "cat2"],
        "num": ["num1", "num2"],
        "cols": ["bin1", "bin2", "cat1", "cat2", "num1", "num2"],
        "binary_indicator": ["1", "yes", "true", "positive", "t", "y"],
        "data_split_idx": {
            "train":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "val":[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "test":[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    }
}

def logistic_regression(
    dataname,
    seed=42,
    dataset_config=None,
    use_optuna=True,
    n_trials=100
):
    print('Running Logistic Regression')

    # 1) Load & preprocess labels
    X, y, categorical_indicator, attribute_names = load_dataset(dataname)
    num_classes = y.nunique()
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))

    # 2) Identify columns
    all_cols = np.array(attribute_names)
    cat_mask = np.array(categorical_indicator, dtype=bool)
    cat_cols = list(all_cols[cat_mask])
    num_cols = list(all_cols[~cat_mask])
    bin_cols = dataset_config.get('bin', []) if dataset_config else []
    cat_cols = [c for c in cat_cols if c not in bin_cols]

    # 3) Split once into train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # 4) Impute & scale numericals
    if num_cols:
        num_imp = SimpleImputer(strategy='median')
        X_trainval[num_cols] = num_imp.fit_transform(X_trainval[num_cols])
        X_test[num_cols]     = num_imp.transform(X_test[num_cols])

        scaler = StandardScaler()
        X_trainval[num_cols] = scaler.fit_transform(X_trainval[num_cols])
        X_test[num_cols]     = scaler.transform(X_test[num_cols])

    # 5) Impute & one-hot encode categoricals
    if cat_cols:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_trainval[cat_cols] = cat_imp.fit_transform(X_trainval[cat_cols])
        X_test[cat_cols]     = cat_imp.transform(X_test[cat_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_trainval[cat_cols])

        def encode_df(df):
            arr = encoder.transform(df[cat_cols])
            cols = encoder.get_feature_names_out(cat_cols)
            return pd.concat([
                df.drop(columns=cat_cols).reset_index(drop=True),
                pd.DataFrame(arr, columns=cols)
            ], axis=1)

        X_trainval = encode_df(X_trainval)
        X_test     = encode_df(X_test)

    # 6) Internal train/val split for tuning
    def train_val_split():
        return train_test_split(
            X_trainval, y_trainval,
            test_size=0.125,
            stratify=y_trainval,
            random_state=seed
        )

    # 7) Objective for Optuna: tune only on validation AUC
    def objective(trial):
        C        = trial.suggest_float("C",        1e-4, 1e2,   log=True)
        penalty  = trial.suggest_categorical("penalty", ["l1", "l2"])
        max_iter = trial.suggest_int("max_iter", 100, 1000, step=100)

        X_tr, X_val, y_tr, y_val = train_val_split()
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="saga",         # saga supports l1, l2, none
            max_iter=max_iter,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)
        if num_classes == 2:
            return roc_auc_score(y_val, y_proba[:, 1])
        else:
            return roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

    # 8) Run tuning (or skip)
    if use_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial.params
        print("Best hyperparameters:", best)
    else:
        best = {"C":1.0, "penalty":"l2", "max_iter":100}

    # 9) Final train on full train+val, then test
    final_model = LogisticRegression(
        C=best["C"],
        penalty=best["penalty"],
        solver="saga",
        max_iter=best["max_iter"],
        random_state=seed,
        n_jobs=-1
    )
    final_model.fit(X_trainval, y_trainval)

    # 10) Evaluate on test set
    y_proba = final_model.predict_proba(X_test)
    y_pred  = y_proba.argmax(axis=1)

    if num_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
        f1  = f1_score(y_test, y_pred, average="binary")
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        f1  = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")
    return auc, f1, acc
    
# def logistic_regression(dataname, seed=42, dataset_config=None):
#     print('Running Logistic Regression')

#     # Load the dataset
#     #dataset = openml.datasets.get_dataset(dataname)
#     X, y, categorical_indicator, attribute_names = load_dataset(dataname)
#     #dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    
#     num_classes = y.nunique()
    
#     # Convert y labels to integers
#     le = LabelEncoder()
#     y = le.fit_transform(y) 
#     y = pd.Series(y)
    
#     # Identify categorical and numerical columns
#     all_cols = np.array(attribute_names)
#     categorical_indicator = np.array(categorical_indicator)
#     cat_cols = list(all_cols[categorical_indicator])
#     num_cols = list(all_cols[~categorical_indicator])

#     if dataset_config is not None:
#         bin_cols = [c for c in cat_cols if c in dataset_config.get('bin', [])]
#     else:
#         bin_cols = []

#     cat_cols = [c for c in cat_cols if c not in bin_cols]
    
#         # Random Train/Test Split using the seed
#     train_dataset, test_dataset, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=seed
#     )

#     # Further split training data into train and validation
#     train_dataset, val_dataset, y_train, y_val = train_test_split(
#         train_dataset, y_train, test_size=0.125, stratify=y_train, random_state=seed
#     )  # 12.5% of train -> validation (so final split: 70% train, 10% val, 20% test)
    
# #     # Assuming X and y are your data and labels
# #     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# #     # Get the indices for the fold
# #     indices = np.arange(X.shape[0])
# #     splits = list(kf.split(indices, y))
# #     train_indices, test_indices = splits[seed]
    

# #     train_dataset = X.iloc[train_indices]
# #     y_train = y.iloc[train_indices]
# #     test_dataset = X.iloc[test_indices]
# #     y_test = y.iloc[test_indices]
    
# #     new_train_indices, val_indices = train_test_split(train_indices, test_size=1/8, stratify=y[train_indices], random_state=seed)
    

# #     new_train_dataset = X.iloc[new_train_indices]
# #     y_new_train = y.iloc[new_train_indices]
# #     val_dataset = X.iloc[val_indices]
# #     y_val = y.iloc[val_indices]

#     # Standardize numerical features
#     if num_cols:
#         num_imputer = SimpleImputer(strategy='median')
#         num_imputer.fit(train_dataset[num_cols])

#         train_dataset[num_cols] = num_imputer.transform(train_dataset[num_cols])
#         test_dataset[num_cols] = num_imputer.transform(test_dataset[num_cols])
        
#         scaler = StandardScaler()
#         scaler.fit(train_dataset[num_cols])
        
#         train_dataset.loc[:, num_cols] = scaler.transform(train_dataset[num_cols])
#         test_dataset.loc[:, num_cols] = scaler.transform(test_dataset[num_cols])
#         val_dataset.loc[:, num_cols] = scaler.transform(val_dataset[num_cols])

#     # One-Hot Encode categorical features
#     if cat_cols:
#         cat_imputer = SimpleImputer(strategy='most_frequent')
#         cat_imputer.fit(train_dataset[cat_cols])

#         train_dataset[cat_cols] = cat_imputer.transform(train_dataset[cat_cols])
#         test_dataset[cat_cols] = cat_imputer.transform(test_dataset[cat_cols])
        
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         encoder.fit(train_dataset[cat_cols])

#         # Transform categorical features
#         train_cat = encoder.transform(train_dataset[cat_cols])
#         val_cat = encoder.transform(val_dataset[cat_cols])
#         test_cat = encoder.transform(test_dataset[cat_cols])

#         # Convert to DataFrame
#         train_cat_df = pd.DataFrame(train_cat, index=train_dataset.index, columns=encoder.get_feature_names_out(cat_cols))
#         val_cat_df = pd.DataFrame(val_cat, index=val_dataset.index, columns=encoder.get_feature_names_out(cat_cols))
#         test_cat_df = pd.DataFrame(test_cat, index=test_dataset.index, columns=encoder.get_feature_names_out(cat_cols))

#         # Drop original categorical columns
#         train_dataset = train_dataset.drop(columns=cat_cols)
#         val_dataset = val_dataset.drop(columns=cat_cols)
#         test_dataset = test_dataset.drop(columns=cat_cols)

#         # Concatenate the one-hot encoded features
#         train_dataset = pd.concat([train_dataset, train_cat_df], axis=1)
#         val_dataset = pd.concat([val_dataset, val_cat_df], axis=1)
#         test_dataset = pd.concat([test_dataset, test_cat_df], axis=1)

#     # Train Logistic Regression Model
#     model = LogisticRegression(
#         max_iter=100,
#         random_state=seed
#     )

#     model.fit(train_dataset, y_train)

#     # Evaluate AUC Score
#     if num_classes == 2:
#         y_pred = model.predict_proba(test_dataset)[:, 1]
#         auc_score = roc_auc_score(y_test, y_pred)
#     else:
#         y_pred = model.predict_proba(test_dataset)
#         auc_score = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")

#     print(f"AUC score: {auc_score}")
#     return auc_score

def xgb(dataname, seed=42, dataset_config=None, use_optuna=True, n_trials=100):
    print('Running XGBoost')
    # from utils import load_dataset

    X, y, categorical_indicator, attribute_names = load_dataset(dataname)
    num_classes = y.nunique()

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = pd.Series(y)

    all_cols = np.array(attribute_names)
    categorical_indicator = np.array(categorical_indicator)
    cat_cols = list(all_cols[categorical_indicator])
    num_cols = list(all_cols[~categorical_indicator])
    bin_cols = [c for c in cat_cols if dataset_config and c in dataset_config.get('bin', [])]
    cat_cols = [c for c in cat_cols if c not in bin_cols]

    for col in cat_cols:
        X[col] = X[col].astype("category")

    # Split data once
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    if num_cols:
        scaler = StandardScaler()
        scaler.fit(X_trainval[num_cols])
        X_trainval[num_cols] = scaler.transform(X_trainval[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    def train_eval(params):
        # Internal split: train/val from trainval
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=seed
        )
        if num_classes == 2:
            model = XGBClassifier(
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                n_estimators=params.get("n_estimators", 100),
                random_state=seed,
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=10,
                enable_categorical=True
            )
        else:
            model = XGBClassifier(
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                n_estimators=params.get("n_estimators", 100),
                random_state=seed,
                use_label_encoder=False,
                eval_metric="mlogloss",
                early_stopping_rounds=10,
                enable_categorical=True
            )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        y_prob = model.predict_proba(X_val)
        y_pred = y_prob.argmax(axis=1)

        if num_classes == 2:
            auc = roc_auc_score(y_val, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")

        return auc

    if use_optuna:
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "learning_rate": trial.suggest_float(
                "learning_rate",
                math.exp(-7),   # ≈0.00091188
                1.0,
                log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 4000)
            }
            return train_eval(params)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print("Best trial params:", study.best_trial.params)
        best_params = study.best_trial.params
    else:
        best_params = {}

    # Final training on entire trainval set with best params
    if num_classes == 2:
        final_model = XGBClassifier(
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.1),
            n_estimators=best_params.get("n_estimators", 100),
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=10,
            enable_categorical=True
        )
    else:
        final_model = XGBClassifier(
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.1),
            n_estimators=best_params.get("n_estimators", 100),
            random_state=seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            early_stopping_rounds=10,
            enable_categorical=True
        )
    final_model.fit(X_trainval, y_trainval, eval_set=[(X_trainval, y_trainval)], verbose=False)

    # Evaluate on test set
    y_prob = final_model.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)

    if num_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    f1 = f1_score(y_test, y_pred, average="binary" if num_classes == 2 else "macro")
    acc = accuracy_score(y_test, y_pred)

    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")
    return auc, f1, acc

# def xgb(dataname, seed=42, dataset_config=None):
#     print('Running XGBoost')

#     # Load the dataset
#     #dataset = openml.datasets.get_dataset(dataname)
#     X, y, categorical_indicator, attribute_names = load_dataset(dataname)
#     #dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    
#     num_classes = y.nunique()
    
#     # Convert y labels to integers
#     le = LabelEncoder()
#     y = le.fit_transform(y) 
#     y = pd.Series(y)
    
#     # Identify categorical and numerical columns
#     all_cols = np.array(attribute_names)
#     categorical_indicator = np.array(categorical_indicator)
#     cat_cols = list(all_cols[categorical_indicator])
#     num_cols = list(all_cols[~categorical_indicator])

#     if dataset_config is not None:
#         bin_cols = [c for c in cat_cols if c in dataset_config.get('bin', [])]
#     else:
#         bin_cols = []

#     cat_cols = [c for c in cat_cols if c not in bin_cols]

#     # Ensure categorical columns are in 'category' dtype
#     for col in cat_cols:
#         X[col] = X[col].astype("category")

#     # Random Train/Test Split using the seed
#     train_dataset, test_dataset, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=seed
#     )

#     # Further split training data into train and validation
#     train_dataset, val_dataset, y_train, y_val = train_test_split(
#         train_dataset, y_train, test_size=0.125, stratify=y_train, random_state=seed
#     )  # 12.5% of train -> validation (so final split: 70% train, 10% val, 20% test)
    
# #     #     # Assuming X and y are your data and labels
# #     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# #     # Get the indices for the fold
# #     indices = np.arange(X.shape[0])
# #     splits = list(kf.split(indices, y))
# #     train_indices, test_indices = splits[seed]
    

# #     train_dataset = X.iloc[train_indices]
# #     y_train = y.iloc[train_indices]
# #     test_dataset = X.iloc[test_indices]
# #     y_test = y.iloc[test_indices]
    
# #     new_train_indices, val_indices = train_test_split(train_indices, test_size=1/8, stratify=y[train_indices], random_state=seed)
    

# #     new_train_dataset = X.iloc[new_train_indices]
# #     y_new_train = y.iloc[new_train_indices]
# #     val_dataset = X.iloc[val_indices]
# #     y_val = y.iloc[val_indices]

#     # Standardize numerical features
#     if num_cols:
#         scaler = StandardScaler()
#         scaler.fit(train_dataset[num_cols])
        
#         train_dataset.loc[:, num_cols] = scaler.transform(train_dataset[num_cols])
#         test_dataset.loc[:, num_cols] = scaler.transform(test_dataset[num_cols])
#         val_dataset.loc[:, num_cols] = scaler.transform(val_dataset[num_cols])

#     # Train XGBoost Classifier with categorical features
#     model = XGBClassifier(
#         n_estimators=10000,
#         random_state=seed,
#         max_depth=6,
#         use_label_encoder=False,
#         eval_metric="mlogloss",
#         early_stopping_rounds=10,
#         enable_categorical=True
#     )

#     model.fit(
#         train_dataset,
#         y_train,
#         eval_set=[(train_dataset, y_train), (val_dataset, y_val)],
#         verbose=True
#     )

#     # Evaluate AUC Score
#     if num_classes == 2:
#         y_pred = model.predict_proba(test_dataset)[:, 1]
#         auc_score = roc_auc_score(y_test, y_pred)
#     else:
#         y_pred = model.predict_proba(test_dataset)
#         auc_score = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")

#     print(f"AUC score: {auc_score}")
#     return auc_score

def load_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):
    '''Load datasets from the local device or from openml.datasets.

    Parameters
    ----------
    dataname: str or int
        the dataset name/index intended to be loaded from openml. or the directory to the local dataset.
    
    dataset_config: dict
        the dataset configuration to specify for loading. Please note that this variable will
        override the configuration loaded from the local files or from the openml.dataset.
    
    encode_cat: bool
        whether encoder the categorical/binary columns to be discrete indices, keep False for TransTab models.
    
    data_cut: int
        how many to split the raw tables into partitions equally; set None will not execute partition.

    seed: int
        the random seed set to ensure the fixed train/val/test split.

    Returns
    -------
    all_list: list or tuple
        the complete dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    train_list: list or tuple
        the train dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    val_list: list or tuple
        the validation dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    test_list: list
        the test dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    cat_col_list: list
        the list of categorical column names.

    num_col_list: list
        the list of numerical column names.

    bin_col_list: list
        the list of binary column names.

    '''
    if dataset_config is None: dataset_config = OPENML_DATACONFIG
    if isinstance(dataname, str):
        # load a single tabular data
        return load_single_data(dataname=dataname, dataset_config=dataset_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
    
    if isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        all_list = []
        train_list, val_list, test_list = [], [], []
        for dataname_ in dataname:
            data_config = dataset_config.get(dataname_, None)
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(dataname_, dataset_config=data_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)
        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list

#def feature_sets(X, data_cut):

def load_dataset(dataname):
    try:
        # Try loading dataset from OpenML
        dataset = openml.datasets.get_dataset(dataname, download_all_files=False)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe', 
            target=dataset.default_target_attribute
        )
        X = X.copy()
        print(f"Dataset '{dataname}' loaded from OpenML.")
    except Exception as e:
        print(f"OpenML dataset '{dataname}' not found. Trying CSV file...")

        # Try loading dataset from CSV file
        csv_file_path = f'{dataname}.csv'  # Adjust path as needed
        df = pd.read_csv(csv_file_path)
        try:
            if dataname == 'stroke':
                # Define the target column and the categorical indicator
                target_column = 'stroke' # Replace with your actual target column name
                categorical_indicator = {
                    'gender': True,               # Binary categorical (Male/Female)
                    'age': False,                 # Numerical (continuous)
                    'hypertension': False,         # Numerical (0/1 indicator, but treated as numeric)
                    'heart_disease': False,        # Numerical (0/1 indicator, but treated as numeric)
                    'ever_married': True,          # Binary categorical (Yes/No)
                    'work_type': True,             # Categorical (e.g., Private, Govt, Self-employed)
                    'Residence_type': True,        # Binary categorical (Urban/Rural)
                    'avg_glucose_level': False,    # Numerical (continuous)
                    'bmi': False,                  # Numerical (continuous)
                    'smoking_status': True         # Categorical (e.g., Never smoked, Former smoker, Smokes)

                }
            
            if dataname == 'seismic_bumps':  
                target_column = 'class'
                categorical_indicator = {
                        'seismic': True,         # Binary categorical
                        'seismoacoustic': True,  # Binary categorical
                        'shift': True,           # Binary categorical
                        'ghazard': True,         # Binary categorical
                        'genergy': False,        # Numerical
                        'gpuls': False,          # Numerical
                        'gdenergy': False,       # Numerical
                        'gdpuls': False,         # Numerical
                        'nbumps': False,         # Numerical (count)
                        'nbumps2': False,        # Numerical (count)
                        'nbumps3': False,        # Numerical (count)
                        'nbumps4': False,        # Numerical (count)
                        'nbumps5': False,        # Numerical (count)
                        'nbumps6': False,        # Numerical (count)
                        'nbumps7': False,        # Numerical (count)
                        'nbumps8': False,        # Numerical (count)
                        'nbumps9': False,        # Numerical (count)
                        'energy': False,         # Numerical
                        'maxenergy': False,      # Numerical
                    }


            # Separate features (X), target (y), and other metadata
            X = df.drop(columns=[target_column])
            y = df[target_column]
            attribute_names = X.columns.tolist()

            # Convert the categorical indicator to a list
            categorical_indicator = [categorical_indicator.get(col, False) for col in X.columns]
            print(f"Dataset '{dataname}' loaded from CSV.")
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file_path}' not found.")
            return None, None, None, None

    return X, y, categorical_indicator, attribute_names

def load_single_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):
    '''Load tabular dataset from local or from openml public database.
    args:
        dataname: Can either be the data directory on `./data/{dataname}` or the dataname which can be found from the openml database.
        dataset_config: 
            A dict like {'dataname':{'bin': [col1,col2,...]}} to indicate the binary columns for the data obtained from openml.
            Also can be used to {'dataname':{'cols':[col1,col2,..]}} to assign a new set of column names to the data
        encode_cat:  Set `False` if we are using transtab, otherwise we set it True to encode categorical values into indexes.
        data_cut: The number of cuts of the training set. Cut is performed on both rows and columns.
    outputs:
        allset: (X,y) that contains all samples of this dataset
        trainset, valset, testset: the train/val/test split
        num_cols, cat_cols, bin_cols: the list of numerical/categorical/binary column names
    '''
    print('####'*10)
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname}')
        filename = os.path.join(dataname, 'data_processed.csv')
        df = pd.read_csv(filename, index_col=0)
        y = df['target_label']
        X = df.drop(['target_label'],axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]

        X.columns = all_cols
        attribute_names = all_cols
        ftfile = os.path.join(dataname, 'numerical_feature.txt')
        if os.path.exists(ftfile):
            with open(ftfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []
        bnfile = os.path.join(dataname, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        cat_cols = [col for col in all_cols if col not in num_cols and col not in bin_cols]

        # update cols by loading dataset_config
        if dataset_config is not None:
            if 'columns' in dataset_config:
                new_cols = dataset_config['columns']
                X.columns = new_cols

            if 'bin' in dataset_config:
                bin_cols = dataset_config['bin']
            
            if 'cat' in dataset_config:
                cat_cols = dataset_config['cat']

            if 'num' in dataset_config:
                num_cols = dataset_config['num']
        
    else:
        
        # dataset = openml.datasets.get_dataset(dataname)
        #print(dataset)
        X,y,categorical_indicator, attribute_names = load_dataset(dataname)
        
        X = X.copy()
        
        '''
        if isinstance(dataname, int):
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            dataname = openml_list.loc[openml_list.did == dataname].name.values[0]
        else:
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            print(f'openml data index: {openml_list.loc[openml_list.name == dataname].index[0]}')
        
        print(f'load data from {dataname}')
        '''
        '''
        
        # Load the dataset from the CSV file
        csv_file_path = f'{dataname}.csv'  # Replace with the path to your CSV file
        df = pd.read_csv(csv_file_path)
        
        target_column = ''
        categorical_indicator = {}
        
        print(dataname)
        
        if dataname == 'credit-g':
        
            # Define the target column and the categorical indicator
            target_column = 'Class' # Replace with your actual target column name
            categorical_indicator = {
                'Status': True,
                'Credit_history': True,
                'Purpose': True,
                'Savings': True,
                'Employment': True,
                'Personal_status': True,
                'Other_debtors': True,
                'Property': True,
                'Other_installment_plans': True,
                'Housing': True,
                'Job': True,
                'Telephone': True,
                'Foreign_worker': True
            }
        if dataname == 'dresses-sales':
            target_column = 'Recommendation'
            categorical_indicator = {
                'Dress_ID': False,  # Likely an identifier
                'Style': True,
                'Price': True,
                'Rating': False,  # Numerical
                'Size': True,
                'Season': True,
                'NeckLine': True,
                'SleeveLength': True,
                'Material': True,
                'FabricType': True,
                'Decoration': True,
                'Pattern Type': True,
            }
        if dataname == 'cylinder-bands':
            target_column = 'blade pressure'
            categorical_indicator = {
                    'Title': True,
                    'Sources': True,
                    'Creator': True,
                    'Donor': True,
                    'Date': True,
                    'Past Usage': True,
                    'Relevant Information': True,
                    'timestamp': True,
                    'cylinder number': True,
                    'customer': True,
                    'job number': False,
                    'grain screened': True,
                    'ink color': True,
                    'proof on ctd ink': True,
                    'blade mfg': True,
                    'cylinder division': False,
                    'paper type': True,
                    'ink type': True,
                    'direct steam': False,
                    'solvent type': True,
                    'type on cylinder': True,
                    'press type': True,
                    'press': False,
                    'unit number': False,
                    'cylinder size': False,
                    'paper mill location': True,
                    'plating tank': False,
                    'proof cut': False,
                    'viscosity': False,
                    'caliper': False,
                    'ink temperature': False,
                    'humidity': False,
                    'roughness': False,
                    'responsibilities require the expert to do what an expert does best': True,
                    'Number of Instances': True,
                    'Number of Attributes': True,
                    'Attribute Information': True,

            }
        if dataname == 'credit-approval':
            # Define the target column and the categorical indicator
            target_column = 'A16' # Replace with your actual target column name
            categorical_indicator = {
                'A1': True,
                'A2': False,
                'A3': True,
                'A4': True,
                'A5': True,
                'A6': True,
                'A7': True,
                'A8': False,
                'A9': True,
                'A10': True,
                'A11': True,
                'A12': True,
                'A13': True,
                'A14': False,
                'A15': False,
            }
            
        if dataname == 'adult':
            # Define the target column and the categorical indicator
            target_column = 'income' # Replace with your actual target column name
            categorical_indicator = {    
                'workclass': True,           # Categorical
                'fnlwgt': False,             # Numerical
                'education': True,           # Categorical
                'education-num': False,      # Numerical
                'marital-status': True,      # Categorical
                'occupation': True,          # Categorical
                'relationship': True,        # Categorical
                'race': True,                # Categorical
                'sex': True,                 # Binary, treated as categorical
                'capital-gain': False,       # Numerical
                'capital-loss': False,       # Numerical
                'hours-per-week': False,      # Numerical
                'gender': True,
                'native-country': True
            }
        if dataname == 'diabetes':
            # Define the target column and the categorical indicator
            target_column = 'Outcome' # Replace with your actual target column name
            categorical_indicator = {    
             'Pregnancies': False,
             'Glucose': False,
             'BloodPressure': False,
             'SkinThickness': False,
             'Insulin': False,
             'BMI': False,
             'DiabetesPedigreeFunction': False,
             'Age': False
            }
        if dataname == 'heart-disease':
            # Define the target column and the categorical indicator
            target_column = 'target' # Replace with your actual target column name
            categorical_indicator = {
                'age': False,                # Numerical (integer)
                'sex': True,                 # Categorical (binary, e.g., Male/Female)
                'cp': True,                  # Categorical
                'trestbps': False,           # Numerical (resting blood pressure)
                'chol': False,               # Numerical (serum cholesterol)
                'fbs': True,                 # Categorical (fasting blood sugar > 120 mg/dl)
                'restecg': True,             # Categorical
                'thalach': False,            # Numerical (maximum heart rate achieved)
                'exang': True,               # Categorical (exercise induced angina)
                'oldpeak': False,            # Numerical (ST depression)
                'slope': True,               # Categorical
                'ca': False,                 # Numerical (number of major vessels)
                'thal': True                 # Categorical
            }
        if dataname == 'bank-marketing':
            # Define the target column and the categorical indicator
            target_column = 'y' # Replace with your actual target column name
            categorical_indicator = {
                'age': False,                # Numerical (integer)
                'job': True,                 # Categorical (type of job)
                'marital': True,             # Categorical (marital status)
                'education': True,           # Categorical (education level)
                'default': True,             # Binary (has credit in default)
                'balance': False,            # Numerical (average yearly balance)
                'housing': True,             # Binary (has housing loan)
                'loan': True,                # Binary (has personal loan)
                'contact': True,             # Categorical (contact communication type)
                'day_of_week': True,         # Categorical (day of last contact)
                'month': True,               # Categorical (month of last contact)
                'duration': False,           # Numerical (last contact duration in seconds)
                'campaign': False,           # Numerical (number of contacts during the campaign)
                'pdays': False,              # Numerical (number of days since last contact)
                'previous': False,           # Numerical (number of contacts before this campaign)
                'poutcome': True,            # Categorical (outcome of previous campaign)
                
            }
         

        # Separate features (X), target (y), and other metadata
        X = df.drop(columns=[target_column])
        y = df[target_column]
        attribute_names = X.columns.tolist()
        
        # Convert the categorical indicator to a list
        categorical_indicator = [categorical_indicator.get(col, False) for col in X.columns]
        '''
        
        # drop cols which only have one unique value
        drop_cols = [col for col in attribute_names if X[col].nunique()<=1]

        all_cols = np.array(attribute_names)
        categorical_indicator = np.array(categorical_indicator)
        cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
        num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
        all_cols = [col for col in all_cols if col not in drop_cols]
        
        print('num: ',num_cols)
        print('cat: ',cat_cols)
        
        if dataset_config is not None:
            if 'bin' in dataset_config: bin_cols = [c for c in cat_cols if c in dataset_config['bin']]
        else: bin_cols = []
        cat_cols = [c for c in cat_cols if c not in bin_cols]

        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y,index=X.index)

    # start processing features
    # process num
    X.replace('?', np.nan, inplace=True)
    if len(num_cols) > 0:
        for col in num_cols: X[col].fillna(X[col].mode()[0])#, inplace=True)
        #X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0])#, inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: X[col].fillna(X[col].mode()[0])#, inplace=True)
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t','present'] else 0).values        
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')

    
    X = X[bin_cols + num_cols + cat_cols]

    # rename column names if is given
    if dataset_config is not None:
        data_config = dataset_config
        if 'columns' in data_config:
            new_cols = data_config['columns']
            X.columns = new_cols
            attribute_names = new_cols

        if 'bin' in data_config:
            bin_cols = data_config['bin']
        
        if 'cat' in data_config:
            cat_cols = data_config['cat']

        if 'num' in data_config:
            num_cols = data_config['num']


    # split train/val/test
    data_split_idx = None
    if dataset_config is not None:
        data_split_idx = dataset_config.get('data_split_idx', None)

    if data_split_idx is not None:
        train_idx = data_split_idx.get('train', None)
        val_idx = data_split_idx.get('val', None)
        test_idx = data_split_idx.get('test', None)

        if train_idx is None or test_idx is None:
            raise ValueError('train/test split indices must be provided together')
    
        else:
            train_dataset = X.iloc[train_idx]
            y_train = y[train_idx]
            test_dataset = X.iloc[test_idx]
            y_test = y[test_idx]
            if val_idx is not None:
                val_dataset = X.iloc[val_idx]
                y_val = y[val_idx]
            else:
                val_dataset = None
                y_val = None
    else:
        # split train/val/test
        
        train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        val_size = int(len(y)*0.1)
        
        # First, split off the 10% test set
        # X_temp, test_dataset, y_temp, y_test = train_test_split(
        #     X, y, test_size=0.1, random_state=seed, stratify=y, shuffle=True
        # )
        
        # # Then, split the remaining 90% into 70% train and 20% val => train = 70/90 ≈ 0.777
        # train_dataset, val_dataset, y_train, y_val = train_test_split(
        #     X_temp, y_temp, test_size=2/9, random_state=seed, stratify=y_temp, shuffle=True
        # )
        
        val_dataset = train_dataset.iloc[-val_size:]
        y_val = y_train[-val_size:]
        train_dataset = train_dataset.iloc[:-val_size]
        y_train = y_train[:-val_size]
        
         
        # Transformation for numerical columns
        if len(num_cols) > 0:
            scaler = MinMaxScaler()
            scaler.fit(train_dataset[num_cols])
            train_dataset[num_cols] = scaler.transform(train_dataset[num_cols])
            val_dataset[num_cols] = scaler.transform(val_dataset[num_cols])
            test_dataset[num_cols] = scaler.transform(test_dataset[num_cols])
        
        '''
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        indices = np.arange(X.shape[0])
        print('For fold: ', seed)
        splits = list(splitter.split(indices, y))
        train_indices, test_indices = splits[seed]
        train_indices, valid_indices = train_test_split(train_indices, test_size=1/8, stratify=y[train_indices], random_state=42)
        
        train_dataset = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        val_dataset = X.iloc[valid_indices]
        y_val = y.iloc[valid_indices]
        test_dataset= X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        '''
    if data_cut is not None:
        np.random.shuffle(all_cols)
        sp_size=int(len(all_cols)/data_cut)
        col_splits = np.split(all_cols, range(0,len(all_cols),sp_size))[1:]
        new_col_splits = []
        for split in col_splits:
            candidate_cols = np.random.choice(np.setdiff1d(all_cols, split), int(sp_size/2), replace=False)
            new_col_splits.append(split.tolist() + candidate_cols.tolist())
        if len(col_splits) > data_cut:
            for i in range(len(col_splits[-1])):
                new_col_splits[i] += [col_splits[-1][i]]
                new_col_splits[i] = np.unique(new_col_splits[i]).tolist()
            new_col_splits = new_col_splits[:-1]

        # cut subset
        trainset_splits = np.array_split(train_dataset, data_cut)
        train_subset_list = []
        for i in range(data_cut):
            train_subset_list.append(
                (trainset_splits[i][new_col_splits[i]], y_train.loc[trainset_splits[i].index])
            )
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X, y), train_subset_list, (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols

    else:
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols