import os
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
)

random.seed(42)

multi_label = None

def save_metrics(result_path: str, data_dict: dict):
    columns = [
        'model_name', 
        'dataset_name',
        'dataset_state',
        'subsampling_rate',
        'batch_size',
        'epoch',
        'learning_rate',
        'text_max_length',
        'weight_decay',

        'semantic_label',
        'stratified_sampling',
        'with_example',
        'alpha',
        'beta',
        'gamma',
        'momentum_rate',
        'cls_loss',
        'centroid',

        'accuracy',

        'micro-precision@1',
        'micro-recall@1',
        'micro-F1@1',
        'macro-precision@1',
        'macro-recall@1',
        'macro-F1@1',
        'hamming-loss@1',

        'micro-precision@3',
        'micro-recall@3',
        'micro-F1@3',
        'macro-precision@3',
        'macro-recall@3',
        'macro-F1@3',
        'hamming-loss@3',

        'micro-precision@5',
        'micro-recall@5',
        'micro-F1@5',
        'macro-precision@5',
        'macro-recall@5',
        'macro-F1@5',
        'hamming-loss@5',

        'micro-precision',
        'micro-recall',
        'micro-F1',
        'macro-precision',
        'macro-recall',
        'macro-F1',
        'hamming-loss',
    ]

    if os.path.isfile(result_path):
        result = pd.read_csv(result_path)
    else:
        result = pd.DataFrame(columns=columns)

    # clean data dict
    for key in list(data_dict.keys()):
        if key.startswith('eval_'):
            new_key = key.lstrip('eval_')
            if new_key not in columns:
                data_dict.pop(key)
            else:
                data_dict[new_key] = data_dict.pop(key)

    result.loc[len(result)] = data_dict
    result.to_csv(result_path, index=False)


def to_one_hot(x: np.ndarray, n_classes: int):
    n_samples = len(x)
    one_hot = np.zeros((n_samples, n_classes), dtype="int")
    for i in range(n_samples):
        for pred_idx in x[i]:
            if 0 <= pred_idx < n_classes:
                one_hot[i][pred_idx] = 1
    return one_hot


def compute_metrics_hits(eval_preds):
    logits, labels = eval_preds
    labels = labels.tolist()
    logits = (-logits).argsort()
    n = len(labels)
    ret = dict()
    for i in [1, 3, 5]:
        recall = logits[:, :i].tolist()
        hit = 0
        for j in range(n):
            for pred in recall[j]:
                if multi_label:
                    if pred in labels[j]:
                        hit += 1
                        break
                else:
                    if pred == labels[j]:
                        hit += 1
                        break
        ret[f"recall@{i}"] = hit / n
    return ret


def compute_metrics(eval_preds, threshold=0.5):
    logits, y_true = eval_preds
    metrics = dict()

    # multi classification
    if y_true.ndim == 1:
        y_pred = logits.argmax(axis=1)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # multi-label classification
    else:
        if isinstance(logits, tuple):
            logits = logits[0]
        n_classes = logits.shape[1]
        if not (y_true.shape[1] == n_classes and ((y_true == 0) | (y_true == 1)).all()):
            y_true = to_one_hot(y_true, n_classes)

        sigmoid = torch.nn.Sigmoid()
        logits = sigmoid(torch.Tensor(logits))

        y_pred = (-logits).argsort()
        for k in [1, 3, 5]:
            y_pred_k = y_pred[:, :k]
            y_pred_k = to_one_hot(y_pred_k, n_classes)
            (
                metrics[f"micro-precision@{k}"],
                metrics[f"micro-recall@{k}"],
                metrics[f"micro-F1@{k}"],
                _,
            ) = precision_recall_fscore_support(
                y_true, y_pred_k, average="micro", zero_division=0.0
            )
            (
                metrics[f"macro-precision@{k}"],
                metrics[f"macro-recall@{k}"],
                metrics[f"macro-F1@{k}"],
                _,
            ) = precision_recall_fscore_support(
                y_true, y_pred_k, average="macro", zero_division=0.0
            )
            metrics[f"hamming-loss@{k}"] = hamming_loss(y_true, y_pred_k)

    # micro-F1 and macro-F1
    y_pred = np.zeros(logits.shape)
    y_pred[np.where(logits >= threshold)] = 1
    (
        metrics["micro-precision"],
        metrics["micro-recall"],
        metrics["micro-F1"],
        _,
    ) = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0.0
    )
    (
        metrics["macro-precision"],
        metrics["macro-recall"],
        metrics["macro-F1"],
        _,
    ) = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0.0
    )
    metrics['hamming-loss'] = hamming_loss(y_true, y_pred)

    return metrics

def subsampling(data: list[dict], subsampling_rate: float):

    # count classes
    train_class_count = defaultdict(int)
    for line in data:
        for label in line['labels']:
            train_class_count[label] += 1

    # collect samples in the same label
    label_collection = [[] for _ in range(len(train_class_count))]
    for line in data:
        idx = line['labels'][0]
        label_collection[idx].append(line)

    # subsampling
    for idx, collection in enumerate(label_collection):
        label_collection[idx] = random.sample(collection, math.ceil(len(collection) * subsampling_rate))

    # shuffle train dataset
    collection_pointer = [0] * len(label_collection)

    shuffle_train = []
    global_idx, idx = 0, 0
    while global_idx < sum(len(t) for t in label_collection):
        if collection_pointer[idx] < len(label_collection[idx]):
            shuffle_train.append(label_collection[idx][collection_pointer[idx]])
            collection_pointer[idx] += 1
            idx = (idx + 1) % len(label_collection)
            global_idx += 1
        else:
            idx = (idx + 1) % len(label_collection)
    
    return shuffle_train
