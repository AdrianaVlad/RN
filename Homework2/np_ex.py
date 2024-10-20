import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])
actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])
predicted = predicted.astype(bool)
actual = actual.astype(bool)

big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn


@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    TP = (gt & pred).sum()
    FP = (pred & ~gt).sum()
    FN = (~pred & gt).sum()
    return TP,FP,FN,len(pred)-TP-FP-FN


assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)


rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)

assert rez_1 == rez_2

@timed(use_seconds=True, show_args=True)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    return ((pred&gt).sum()+(~pred&~gt).sum())/len(pred)


assert accuracy_sklearn(actual, predicted) == accuracy_numpy(actual, predicted)

rez_1 = accuracy_sklearn(big_actual, big_predicted)
rez_2 = accuracy_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)

@timed(use_seconds=True, show_args=True)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    f1 = f1_score(gt, pred)
    return np.round(f1, decimals=6)

@timed(use_seconds=True, show_args=True)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    tp = (pred&gt).sum()
    precision = tp/pred.sum()
    recall = tp/gt.sum()
    f1=2*precision*recall/(precision+recall)
    return np.round(f1, decimals=6)

assert f1_score_sklearn(actual, predicted) == f1_score_numpy(actual, predicted)

rez_1 = f1_score_sklearn(big_actual, big_predicted)
rez_2 = f1_score_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)