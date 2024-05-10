import logging
from warnings import warn

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from beartype.typing import Collection, Dict, Iterable, List, Optional, Tuple, Union
from pandas import DataFrame
from scipy.stats import entropy as kulback_liebler_divergence
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import Tensor
from torch.nn import MSELoss

from mltoolkit.tensors import arr_to_tensor, threshold_confidences

logger = logging.getLogger(__name__)


# do not reinstantitate
mse_loss = MSELoss(reduction="mean")
# bce_loss = BCELoss(reduction="none")


@torch.no_grad()
@beartype
def calc_metrics(
    y_pred: Union[np.ndarray, Tensor],
    y_true: Union[np.ndarray, Tensor],
    threshold: float,
    class_labels: Optional[Collection] = None,
) -> Dict[str, float]:
    """
    Entry point for computing all metrics during model training.

    note: do not compute loss in here, or PyTorch will complain about call numpy() on
    Variable that requires grad.
    """
    assert len(y_pred) > 0
    assert len(y_pred) == len(y_true)
    assert 0.1 <= threshold < 1, "Due to label smoothing, threshold must be >= 0.1"

    all_metrics = {}
    y_pred = threshold_confidences(y_pred, threshold)
    y_true = threshold_confidences(y_true, threshold)

    class_labels = _process_class_labels(y_pred, class_labels=class_labels)

    # macro avg (unweighted simple avg)
    avg_precision, avg_recall, avg_f1, _supt = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, beta=1.0, average="macro", zero_division=0
    )

    # calculate macro f0.5, f2
    for metric, beta in [("avg_f0.5", 0.5), ("avg_f2", 2.0)]:
        score = fbeta_score(
            y_true=y_true, y_pred=y_pred, beta=beta, average="macro", zero_division=0
        )
        all_metrics[metric] = score
    # calculate calibrated f1 score

    equal_vals = np.equal(y_pred, y_true)
    # for each sample, test if we got the prediction completely correct. then, avg across samples
    # this is a stricter measure of accuracy than the simple avg of the per-class accuracies
    avg_full_match = equal_vals.all(axis=1).mean()

    accuracies = equal_vals.mean(axis=0)
    avg_accuracy = accuracies.mean()

    all_metrics.update(
        dict(
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_accuracy=avg_accuracy,
            avg_full_match=avg_full_match,  # 2022-11-30: renamed full_match to avg_full_match
        )
    )

    # per class metrics
    precisions, recalls, f1s, _supt = precision_recall_fscore_support(
        y_true, y_pred, beta=1.0, average=None, zero_division=0
    )
    for cls, p, r, f, a in zip(class_labels, precisions, recalls, f1s, accuracies):
        all_metrics.update(
            {
                f"{cls}/precision": p,
                f"{cls}/recall": r,
                f"{cls}/f1": f,
                f"{cls}/accuracy": a,
            }
        )

    # calculate micro f0.5, f2
    for metric, beta in [("f0.5", 0.5), ("f2", 2.0)]:
        per_class_f_score = fbeta_score(
            y_true=y_true, y_pred=y_pred, beta=beta, average=None, zero_division=0
        )
        all_metrics.update(
            {
                f"{cls}/{metric}": score
                for cls, score in zip(class_labels, per_class_f_score)
            }
        )

    return all_metrics


@beartype
def calc_rmse(y_true: Iterable, y_pred: Iterable) -> float:
    """
    Calculate the root mean square error between two vectors.
    """
    assert len(y_pred) > 0
    assert len(y_pred) == len(y_true)
    y_pred, y_true = arr_to_tensor(y_pred), arr_to_tensor(y_true)

    rmse = mse_loss(y_pred, y_true)
    return rmse.item()  # Tensor -> float


@beartype
def calc_dkl(
    y_true: Union[np.ndarray, Tensor], y_pred: Union[np.ndarray, Tensor]
) -> Union[float, np.float32, np.ndarray]:
    # check if any row is full of zeros
    if np.any(np.all(np.isclose(y_true, 0), axis=-1)):
        warn(
            "True annotations should not be all zeros; KL-div will "
            "normalize this into a probability distribution, "
            "causing unexpected results. Returning NaN.",
            UserWarning,
        )
        return np.nan
    if np.any(np.all(np.isclose(y_pred, 0), axis=-1)):
        warn(
            "Model predicted all zeros; KL-div will normalize this into a uniform probability distribution.",
            UserWarning,
        )

    # P = ground truth, Q = predictions
    loss = np.mean(
        kulback_liebler_divergence(y_true + 1e-15, qk=y_pred + 1e-15, axis=-1)
    )
    return loss


@beartype
def calc_areas_under_curves(
    y_pred: Union[np.ndarray, Tensor], y_true: Union[np.ndarray, Tensor]
) -> Dict[str, float]:
    """
    Compute areas under curves.
    """
    assert len(y_pred) > 0
    assert len(y_pred) == len(y_true)

    all_metrics = {}
    # we need to ensure that the y_true labels are hard labels. they could be soft labels if we are using label
    # smoothing, or using fuzzy targets.
    y_true = threshold_confidences(y_true, thresh=0.5)

    class_names = _process_class_labels(y_true)

    # macro avg (unweighted simple avg)
    # using one-vs-rest
    try:
        all_metrics["avg_AUROC"] = roc_auc_score(
            y_true=y_true, y_score=y_pred, average="macro", multi_class="ovr"
        )
    except ValueError:
        logger.warning(
            "AUC score uindefined! We should have minimum 1 positive and 1 negative sample for each class,"
            " or else the whole metric is undefined. This warning extends to mAP scores, too."
        )
        all_metrics["avg_AUROC"] = np.nan
    # AP is roughly equal to AUPRC. it is also known as mAP score, but we're calling it avg_AP to make it easier to
    # discover this metric using Tensorboard regex search
    all_metrics["avg_AP"] = average_precision_score(
        y_true=y_true, y_score=y_pred, average="macro"
    )
    # fixme: make average_percision_calibrated work with multilabels
    # ValueError: multilabel-indicator format is not supported
    # all_metrics["avg_AUPRGC"] = average_precision_calibrated(y_true=y_true, y_pred=y_pred, average="macro")

    # per class metrics
    try:
        roc_aucs = roc_auc_score(y_true=y_true, y_score=y_pred, average=None)
    except ValueError:
        # see try/except block above
        roc_aucs = [np.nan] * len(class_names)

    aps = average_precision_score(y_true=y_true, y_score=y_pred, average=None)
    # auPRGCs = average_precision_calibrated(y_true=y_true, y_pred=y_pred, average=None)
    for cls, roc_auc, ap in zip(class_names, roc_aucs, aps):
        all_metrics.update({f"{cls}/AUROC": roc_auc, f"{cls}/AP": ap})

    return all_metrics


@beartype
def compute_multilabel_confusion_matrix(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    threshold: float,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix for multi-label classification.

    Returns:
        confusion_matrix: np.ndarray, shape=(num_classes, 4)
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    y_pred = (y_pred > threshold).astype(int)
    # returns (n_classes x 2 x 2) nd-array, one 2x2 for each class
    conf_mat = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
    # reshape matrix to N_class x 4
    conf_mat = np.reshape(conf_mat, (num_classes, 4))

    return conf_mat


@beartype
def compute_labeled_dataset_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    threshold: float,
    class_labels: List,
) -> Tuple[DataFrame, float, DataFrame]:
    """
    Compute metrics on all labeled data.

    Note: this is used in the validation notebook
    """
    num_classes = len(class_labels)
    if num_classes <= 1:
        raise ValueError(f"metrics only work w >1 classes. found {num_classes} classes")
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert y_true.index.equals(
        y_pred.index
    ), "y_true and y_pred must have the same index"

    class_labels = _process_class_labels(y_pred, class_labels)
    """
    MSE.
    """
    # TODO: use actual loss function used, such as bce or asl
    mse_per_sample = ((y_true.to_numpy() - y_pred.to_numpy()) ** 2).mean(axis=1)
    # make sure there are no nans
    assert not np.isnan(
        mse_per_sample
    ).any(), f"found {np.sum(np.isnan(mse_per_sample))} nans in mse_per_sample"

    # get top 0.5% highest MSE loss imgs
    # https://stackoverflow.com/a/23734295/4212158
    top_frac = 0.005
    num_samp = int(len(mse_per_sample) * top_frac)
    top_loss_idx = np.argpartition(mse_per_sample, -num_samp)[-num_samp:]
    top_loss_preds = y_pred.iloc[top_loss_idx]
    top_loss = pd.DataFrame(
        index=top_loss_preds.index, data={"mse": mse_per_sample[top_loss_idx]}
    )  # convert to df
    top_loss.sort_values(by="mse", ascending=False, inplace=True)  # highest loss first
    assert top_loss.max().item() == np.max(
        mse_per_sample
    ), "top loss max is not the same as the max of the mse_per_sample"
    assert (
        top_loss.iloc[0].item() == top_loss.max().item()
    ), "top loss max is not the same as the first row"

    mse_macro = mse_per_sample.mean().item()
    """
    Compute Confusion Matrix on full supervised learning dataset.
    """
    # TODO: get class names from the dfs directly instead of having to pass in training_class_ids
    conf_mat = compute_multilabel_confusion_matrix(
        y_true, y_pred, threshold, num_classes
    )
    conf_mat = pd.DataFrame(
        conf_mat,
        columns=["True Negative", "False Positive", "False Negative", "True Positive"],
        index=class_labels,
    )
    # add back in classes col
    conf_mat.insert(0, "Class", value=class_labels)
    # drop T- col as it's not super useful. TODO slice out from npy directly
    # conf_mat.drop(columns=["True Negative"], inplace=True)
    confusion_matrix = conf_mat

    return confusion_matrix, mse_macro, top_loss


@beartype
def _process_class_labels(
    pred_arr: Union[pd.DataFrame, np.ndarray, Tensor],
    class_labels: Optional[Iterable] = None,
) -> Iterable:
    num_classes = pred_arr.shape[1] if pred_arr.ndim > 1 else 1
    if num_classes <= 1:
        raise ValueError(f"metrics only work w >1 classes. found {num_classes} classes")

    if class_labels is None:
        if isinstance(pred_arr, pd.DataFrame):
            class_labels = pred_arr.columns.tolist()
        else:
            class_labels = np.arange(num_classes)

    assert len(class_labels) == num_classes

    return class_labels


@beartype
def slow_calc_accuracy(
    fn: Union[np.int64, int],
    fp: Union[np.int64, int],
    tn: Union[np.int64, int],
    tp: Union[np.int64, int],
) -> float:
    """
    A slow implementation to compare optimized sklearn versions against as a sanity check.
    """
    assert all([x >= 0 for x in (fn, fp, tn, fp)])

    if (tp + tn + fn + fp) == 0:
        accuracy = 0.0
    else:
        accuracy = (tp + tn) / (tp + tn + fn + fp)
    return accuracy


@beartype
def slow_calc_recall(fn: Union[np.int64, int], tp: Union[np.int64, int]) -> float:
    """
    A slow implementation to compare optimized sklearn versions against as a sanity check.
    """
    if tp + fn == 0:  # n_pos = 0
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    return recall


@beartype
def slow_calc_fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    A slow implementation to compare optimized sklearn versions against as a sanity check.
    """

    assert beta > 0

    try:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    except ZeroDivisionError:
        f_beta = 0.0

    if np.isnan(f_beta):
        f_beta = 0.0  # numpy won't throw exception

    return f_beta


@beartype
def slow_calc_precision(fp: Union[np.int64, int], tp: Union[np.int64, int]) -> float:
    """
    A slow implementation to compare optimized sklearn versions against as a sanity check.
    """
    assert fp >= 0
    assert tp >= 0

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    return precision
