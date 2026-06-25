import os
os.chdir('../')
import numpy as np
import pandas as pd
import torch
import time
import optuna
import json
import warnings
import argparse
import cattle
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

def get_cache_path(dataset_name, seed=None):
    """Create unique cache path for dataset and seed."""
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    return os.path.join("cached_datasets", f"{dataset_name}{seed_suffix}.pkl")

def load_cached_data(dataset_name, seed=None):
    """
    Loads dataset from cache if available; otherwise loads from OpenML and caches it.
    Returns: trainset, valset, testset, cat_cols, num_cols, bin_cols
    """
    os.makedirs("cached_datasets", exist_ok=True)
    cache_path = get_cache_path(dataset_name, seed)

    # Check if cache exists
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loaded cached dataset: {dataset_name} (seed={seed})")
            return pickle.load(f)

    # Else, load from OpenML
    print(f"Downloading dataset: {dataset_name} from OpenML...")
    if seed is not None:
        result = cattle.load_data([dataset_name], seed=seed)
    else:
        result = cattle.load_data([dataset_name])

    # Cache the result
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
        print(f"Cached dataset: {dataset_name} at {cache_path}")

    return result
    
def pretrain_source_domain(source, folds, epoch):
    for fold in folds:
        print(f"\nPretraining on seed {fold} (source={source})")
        cattle.random_seed(fold)
        save_path = f'./ckpt/cattle-self-supervised/{source}/{fold}/pretrained/'
        os.makedirs(save_path, exist_ok=True)

        # load all three splits
        _, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_cached_data(source, seed=fold)
        X_train, y_train = trainset[0]
        X_val,   y_val   = valset[0]
        X_test,  y_test  = testset[0]

        # merge for pretraining
        X_all = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
        y_all = pd.concat([y_train, y_val, y_test], axis=0).reset_index(drop=True)

        model = cattle.MaskedClassifier(
            cat_cols, num_cols, bin_cols,
            num_class=len(np.unique(y_all)), p=1
        )
        model = model.to('cuda:0').to(torch.float32)

        # use epoch parameter for pretraining
        model.pretrain_masked(X_all, num_epochs=2, batch_size=128, lr=1e-4)
        model.save(save_path)


def objective(trial, fold, source, target, epoch):
    cattle.random_seed(fold)
    save_path = f'./ckpt/cattle-self-supervised/{source}/{fold}/pretrained/'
    _, src_trainset, _, _, src_cat, src_num, src_bin = cattle.load_data([source])
    _, y_src_train = src_trainset[0]

    # load target data
    _, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_cached_data(target, seed=fold)
    X_train, y_train = trainset[0]
    X_val,   y_val   = valset[0]
    num_class_target = len(np.unique(y_train))

    model = cattle.MaskedClassifier(
        cat_cols, num_cols, bin_cols,
        num_class=len(np.unique(y_train)),
        num_layer=5,
        p=1
    )
    model.load_pretrained_weights(save_path)
    mapping = {0: 4, 1: 4}
    model.apply_ca(mapping=mapping, reset_model=True)
    
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dropout_prob = trial.suggest_categorical("hidden_dropout_prob", [0.0, 0.1, 0.2, 0.3, 0.4])
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.01, 0.05, 0.1])

    args = {
        'num_epoch': epoch,
        'eval_metric': 'auc',
        'eval_less_is_better': False,
        'output_dir': f'./checkpoint/cattle-self-supervised/{source}_{target}',
        'batch_size': batch_size,
        'lr': lr,
        'num_class': num_class_target,
        'save_best': True,
        'exp': f'optuna_{target}_fold{fold}',
        'weight_decay': weight_decay,
        'warmup_ratio' : warmup_ratio,
    }

    # train on train, validate on val
    cattle.train(model, (X_train, y_train), (X_val, y_val), **args)
    y_pred = cattle.predict(model, X_val, y_val)

    # compute validation metrics
    if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 2):
        y_score = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
        auc = roc_auc_score(y_val, y_score)
        y_cls = (y_score > 0.5).astype(int)
        acc = accuracy_score(y_val, y_cls)
        f1  = f1_score(y_val, y_cls)
    else:
        auc = roc_auc_score(y_val, y_pred, multi_class="ovr", average="macro")
        y_cls = np.argmax(y_pred, axis=1)
        acc   = accuracy_score(y_val, y_cls)
        f1    = f1_score(y_val, y_cls, average="macro")

    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("f1_score", f1)
    return auc

def evaluate_per_fold_best(source, target, folds, epoch, n_trials):
    results = {}
    for fold in folds:
        print(f"\nRunning Optuna for fold {fold}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda tr: objective(tr, fold, source, target, epoch), n_trials=n_trials)

        best = study.best_trial
        results[fold] = {
            'val_auc':      best.value,
            'val_accuracy': best.user_attrs["accuracy"],
            'val_f1':       best.user_attrs["f1_score"],
            'best_params':  best.params
        }
        print(f"Fold {fold}: val_AUC={best.value:.4f}, val_ACC={best.user_attrs['accuracy']:.4f}, val_F1={best.user_attrs['f1_score']:.4f}, params={best.params}")

        _, src_trainset, _, _, src_cat, src_num, src_bin = load_cached_data(source)
        _, y_src_train = src_trainset[0]
    
        # final test evaluation
        _, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_cached_data(target, seed=fold)
        X_train, y_train = trainset[0]
        X_val,   y_val   = valset[0]
        X_test,  y_test  = testset[0]


        model = cattle.MaskedClassifier(
            cat_cols, num_cols, bin_cols,
            num_class=len(np.unique(y_train)),
            num_layer=5,
            p=1
        )
        model.load_pretrained_weights(f'./ckpt/cattle-self-supervised/{source}/{fold}/pretrained/')
        mapping = {0: 4, 1: 4}
        model.apply_ca(mapping=mapping, reset_model=True)


        final_args = {
            'num_epoch': epoch,
            'eval_metric': 'auc',
            'eval_less_is_better': False,
            'output_dir': f'./checkpoint/{source}_{target}',
            'batch_size': best.params['batch_size'],
            'lr': best.params['lr'],
            'num_class': len(np.unique(y_train)),
            'save_best': True,
            'exp': f'final_{target}_fold{fold}',
            'weight_decay': best.params['weight_decay'],
            'warmup_ratio' : best.params['warmup_ratio']
        }
        
        cattle.train(model, (X_train, y_train), (X_val, y_val), **final_args)
        
        y_test_pred = cattle.predict(model, X_test, y_test)

        if y_test_pred.ndim == 1 or (y_test_pred.ndim == 2 and y_test_pred.shape[1] == 2):
            y_score = y_test_pred if y_test_pred.ndim == 1 else y_test_pred[:, 1]
            test_auc = roc_auc_score(y_test, y_score)
            y_cls = (y_score > 0.5).astype(int)
            test_acc = accuracy_score(y_test, y_cls)
            test_f1  = f1_score(y_test, y_cls)
        else:
            test_auc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="macro")
            y_cls = np.argmax(y_test_pred, axis=1)
            test_acc = accuracy_score(y_test, y_cls)
            test_f1  = f1_score(y_test, y_cls, average="macro")

        results[fold].update({'test_auc': test_auc, 'test_accuracy': test_acc, 'test_f1': test_f1})
        print(f"Fold {fold}: test_AUC={test_auc:.4f}, test_ACC={test_acc:.4f}, test_F1={test_f1:.4f}")

    # summary statistics across folds (validation)
    aucs = [v['test_auc'] for v in results.values()]
    accs = [v['test_accuracy'] for v in results.values()]
    f1s  = [v['test_f1'] for v in results.values()]
    summary = {
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
        'f1_mean':  np.mean(f1s),  'f1_std':  np.std(f1s)
    }

    print("\nOverall Validation Summary:")
    print(f"  AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  ACC: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"  F1:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Train & fine-tune a CA transformer on tabular data"
    )
    parser.add_argument('--source',        type=str,   required=True)
    parser.add_argument('--target',        type=str,   required=True)
    parser.add_argument('--folds',         type=int,   nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--epoch',         type=int,   default=150)
    parser.add_argument('--trials',        type=int,   default=100)
    parser.add_argument('--skip-pretrain', action='store_true', help='If set, skip the source-domain pretraining step')
    args = parser.parse_args()

    os.chdir(os.path.dirname(__file__))

    if not args.skip_pretrain:
        pretrain_source_domain(args.source, args.folds, args.epoch)
    else:
        print("Skipping source pretraining as requested.")

    results, summary = evaluate_per_fold_best(args.source, args.target, args.folds, args.epoch, args.trials)

    out_dir = f"./self_supervised_results/{args.source}_{args.target}/"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'final_results.json'), 'w') as fp:
        json.dump({'folds': results, **summary}, fp, indent=2)

if __name__ == "__main__":
    main()
