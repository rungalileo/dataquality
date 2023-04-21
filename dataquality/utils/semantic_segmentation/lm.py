from typing import List

import numpy as np
import sentry_sdk
from rungalileo.schemas.vaex import VaexColumn as VC
from rungalileo.utils.constants import ITER_CHUNK_SIZE
from rungalileo.utils.vaex.decorators import copy_dataframes
from vaex.dataframe import DataFrame

# Self confidence is the confidence in a prediction of its given GT label
# This is DIFFERENT than confidence, which is the confidence in the highest probability
# prediction. Can be thought of as ground truth confidence

SELF_CONFIDENCE = "self_confidence"
# Likely mislabeled by confidence
MIS_BY_CLASS = "mislabeled_by_class_confidence"
# Likely mislabeled by noise rate
MIS_BY_NOISE = "mislabeled_by_noise_rate"


def fill_confident_counts(
    probs: np.ndarray,
    gold_idxs: np.ndarray,
    per_class_threshold: np.ndarray,
    confident_counts: np.ndarray,
) -> np.ndarray:
    """Constructs the confident count matrix

    The confident count matrix counts for each pair of labels (i, j) how many
    samples were labeled i but confidently predicted j.

    This is a helper function to partially fill the confident_counts based on the
    batch of probs and gold_idxs. Note this function is fully vectorized!

    :param probs: The batch of probability vectors
    :param gold_idxs: The corresponding batch of gold (given) GT label indexes
    :param per_class_threshold: The average probabilities at a class for
        samples labeled as that class. That is to say, for all samples with the given
        GT label 3, what was their average confidence in prediction 3 (even if 3 wasn't
        their highest probability prediction).
        See utils/vaex/mislabeled.py::_calculate_self_confidence
    :param confident_counts: The zeroed (or partially filled) confident counts to fill
        in this function.
    """
    probs = probs.copy()
    # Boolean mask - for each sample the classes whose probs
    # are higher than their threshold
    higher_than_threshold_mask = probs >= per_class_threshold
    # Mask out classes < threshold
    probs[~higher_than_threshold_mask] = -1
    # Only consider the samples with >= 1 above threshold class
    one_or_more_classes = np.where(np.sum(higher_than_threshold_mask, axis=1) >= 1)[0]

    # Index for the highest confident class above class threshold
    js = np.argmax(probs[one_or_more_classes, :], axis=1)
    gold_idxs = gold_idxs[one_or_more_classes]

    np.add.at(confident_counts, (gold_idxs, js), 1)

    return confident_counts


def normalize_confident_counts(
    confident_counts: np.ndarray, class_counts: np.ndarray
) -> np.ndarray:
    """Normalize the confident_counts -> confidence joint.

    Going from a count matrix to a valid probability matrix by making sure that the rows
    sum to the class distribution. In case this batch doesnt have all classes
    represented, fill with 0

    class_counts is the count of number of samples per unique class in the data
    """
    # If a row got no samples, add a 1 on their diagonal (pretend there is no noise).
    labels_with_no_samples = np.where(confident_counts.sum(axis=1) == 0)[0]
    if len(labels_with_no_samples) > 0:
        partial_diag = np.zeros((len(class_counts), len(class_counts)), dtype=np.int64)
        partial_diag[labels_with_no_samples, labels_with_no_samples] = 1
        confident_counts += partial_diag

    # Normalize each row so that they represent the distribution of labels.
    confidence_joint = (
        confident_counts
        * np.expand_dims(class_counts, axis=1)
        / (confident_counts.sum(axis=1, keepdims=True) + 1e-10)
    )
    # Normalize the entire matrix to have a distribution.
    confidence_joint /= confidence_joint.sum() + 1e-10

    # This accounts for the unlikely scenario that class counts is 0.
    if np.isclose(confidence_joint.sum(), 0.0):
        np.fill_diagonal(confidence_joint, 1.0 / confidence_joint.shape[0])

    return confidence_joint


def mislabeled_by_noise_rate_for_class(
    confidence_joint: np.ndarray,
    class_probs: np.ndarray,
    given_gold_idx: int,
    num_samples: int,
) -> List[int]:
    """Get mislabeled samples for the given class index via noise rate

    See the CL paper for more information, but using the confidence
    joint we can get an estimated probability for the percentage of samples given the i
    GT label but actually belonging to the j GT label. We extrapolate from the
    probability a number of samples that are most likely to be mislabeled for each cell
    in the CJ matrix and choose that many of the highest margin samples
    (highest margin between predicted to be class j but given class i).

    :param confidence_joint: The normalized confidence counts
    :param class_probs: The probability matrix only for samples in the given class index
    :param given_gold_idx: The given gold index being examined
    :param num_samples: The *total* number of samples in the data (not this class)
    """
    mislabeled_idxs_in_class: List[int] = []
    num_classes = class_probs.shape[1]
    # "actual" or "expected" gold index, looking at each class one by one
    for actual_gold_idx in range(num_classes):  # j in the paper
        if given_gold_idx == actual_gold_idx:
            continue

        # number of mislabeled samples for this given / actual GT pair
        num_mislabeled = round(
            confidence_joint[given_gold_idx, actual_gold_idx] * num_samples
        )
        if not num_mislabeled:
            continue

        # Margin is how much more confident the model is in a non-given GT label
        class_samples_margin = (
            class_probs[:, actual_gold_idx] - class_probs[:, given_gold_idx]
        )
        # We want the top `num_mislabeled` samples with the largest margin
        # for the actual/expected class relative to the given class
        mislabeled_idxs = np.argsort(class_samples_margin)[-num_mislabeled:]
        mislabeled_idxs_in_class.extend(mislabeled_idxs)
    return mislabeled_idxs_in_class


@sentry_sdk.trace
@copy_dataframes
def _get_per_class_threshold(df: DataFrame, num_classes: int) -> np.ndarray:
    """Gets the average self-confidence per class

    average self-confidence: The average probabilities at a class for
        samples labeled as that class
    """
    selection = []
    for idx in range(num_classes):
        df.select(f"{VC.gold}=={idx}", name=f"{VC.gold}_{idx}")
        selection.append(f"{VC.gold}_{idx}")
    return df[SELF_CONFIDENCE].mean(selection=selection)


@sentry_sdk.trace
def _get_confident_counts(
    df: DataFrame, per_class_threshold: np.ndarray, num_classes: int
) -> np.ndarray:
    """Gets the confident counts per class

    For each class, counts the number of samples where the confidence is more than
    the threshold of that class
    """
    confident_counts = np.zeros((num_classes, num_classes))
    for i1, i2, chunk in df.evaluate_iterator(
        [VC.prob, VC.gold], chunk_size=ITER_CHUNK_SIZE
    ):
        probs, gold_idxs = chunk
        confident_counts = fill_confident_counts(
            probs, gold_idxs, per_class_threshold, confident_counts
        )

    return confident_counts


@sentry_sdk.trace
def _get_confidence_joint(df: DataFrame) -> np.ndarray:
    """Calculates and normalizes the confident_counts into a confidence_joint"""
    # This is in format {gold_idx: count}
    class_count_dict = df[VC.gold].value_counts().sort_index().to_dict()
    # If there are any missing classes we want to fill them with 0s
    num_classes = df[VC.prob].shape[1]
    count_per_class = np.array(
        [class_count_dict.get(class_idx, 0) for class_idx in range(num_classes)]
    )
    per_class_threshold = _get_per_class_threshold(df, num_classes)
    confident_counts = _get_confident_counts(df, per_class_threshold, num_classes)
    confidence_joint = normalize_confident_counts(confident_counts, count_per_class)
    return confidence_joint


@sentry_sdk.trace
@copy_dataframes
def _get_mislabeled_by_class_confidence(
    df: DataFrame, confidence_joint: np.ndarray
) -> DataFrame:
    """Gets the most likely mislabeled samples per class using the confidence_joint

    See the CL paper for more information, but using the confidence joint we can get
    an estimated probability for the percentage of a class that's likely mislabeled.
    We extrapolate from the probaiblity a number of samples that are most likely to
    be mislabeled for each class and choose that many of the lowest self-confidence
    samples.

    (1) get self confidence for all samples, (2) sort ascending across all, (3) get
    number_of_samples_in_this_class_likely_mislabelled per class and (4) take the top
    number_of_samples_in_this_class_likely_mislabelled per class with the lowest
    self confidence
    """
    num_samples, num_classes = df[VC.prob].shape
    df = df.sort(by=SELF_CONFIDENCE, ascending=True)
    # Number of samples in each class likely to be mislabeled
    num_mislabeled_per_class = np.rint(
        (confidence_joint.sum(axis=1) - confidence_joint.diagonal()) * num_samples
    ).astype("int")

    mislabeled_ids = []
    for class_idx, num_mislabeled in zip(range(num_classes), num_mislabeled_per_class):
        mislabeled_df = df[df[VC.gold] == class_idx]
        # Can't slice an empty dataframe https://github.com/vaexio/vaex/issues/2123
        if num_mislabeled == 0 or len(mislabeled_df) == 0:
            continue
        mislabeled_ids.extend(mislabeled_df[:num_mislabeled][VC.id].tolist())

    df[MIS_BY_CLASS] = df.func.where(df[VC.id].isin(mislabeled_ids), True, False)
    return df


@sentry_sdk.trace
@copy_dataframes
def _get_mislabeled_by_noise_rate(
    df: DataFrame, confidence_joint: np.ndarray
) -> DataFrame:
    """Gets the most likely mislabeled samples per given GT and actual GT pair using the

    confidence_joint. See the CL paper for more information, but using the confidence
    joint we can get an estimated probability for the percentage of samples given the i
    GT label but actually belonging to the j GT label. We extrapolate from the
    probability a number of samples that are most likely to be mislabeled for each cell
    in the CJ matrix and choose that many of the highest margin samples
    (highest margin between predicted to be class j but given class i).
    """
    num_samples, num_classes = df[VC.prob].shape

    mislabeled_ids = []
    for given_gold_idx in range(num_classes):  # i in the paper
        class_df = (df[df[VC.gold] == given_gold_idx]).extract()
        if len(class_df) == 0:
            continue
        class_sample_ids = class_df[VC.id].to_numpy()
        # This numpy matrix is of shape (num_samples_in_this_class, num_classes)
        # This should on average be len(df) floats. This is because
        # each class has on avg num_samples/num_classes samples,
        # and the matrix is that value X num_classes, so on average its
        # ((num_samples/num_classes) * num_classes) == num_samples float values
        # but can potentially be really large if class distribution is very skewed
        # (num_samples X num_classes)
        class_probs = class_df[VC.prob].to_numpy()
        mislabeled_idxs = mislabeled_by_noise_rate_for_class(
            confidence_joint, class_probs, given_gold_idx, num_samples
        )
        mislabeled_ids.extend(class_sample_ids[mislabeled_idxs])

    df[MIS_BY_NOISE] = df.func.where(df[VC.id].isin(mislabeled_ids), True, False)
    return df


@sentry_sdk.trace
@copy_dataframes
def _calculate_self_confidence(df: DataFrame) -> DataFrame:
    """Calculates the self confidence for each row (sample)

    Self confidence is the confidence in a prediction of its given GT label
    This is DIFFERENT than confidence, which is the confidence in the highest
    probability prediction. Can be thought of as ground truth confidence
    """
    df[SELF_CONFIDENCE] = df[VC.prob].column_index(df[VC.gold])
    return df


@sentry_sdk.trace
@copy_dataframes
def get_mislabeled_samples(df: DataFrame) -> DataFrame:
    """Adds a new column "likely_mislabeled" for all samples

    Signifies if a sample is likely to be mislabeled. See rungalileo.ml_core.mislabeled
    """
    df = _calculate_self_confidence(df)
    confidence_joint = _get_confidence_joint(df)
    df = _get_mislabeled_by_class_confidence(df, confidence_joint)
    df = _get_mislabeled_by_noise_rate(df, confidence_joint)
    # Only samples that are mislabeled by both confidence and noise rate are considered
    df[VC.likely_mislabeled] = df.func.where(
        (df[MIS_BY_NOISE]) & (df[MIS_BY_CLASS]), True, False
    )
    tmp_cols = [MIS_BY_CLASS, MIS_BY_NOISE, SELF_CONFIDENCE]
    good_cols = [c for c in df.get_column_names() if c not in tmp_cols]
    return df[good_cols]


import torch


def _calculate_self_confidence(probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Gets the self confidence for each sample meaning the
    confidence in a prediction of its given GT label


    Args:
        probs (torch.Tensor): probability mask for each sample
        gt (torch.Tensor): ground truth label for each sample

    Returns:
        np.ndarray: self confidence for each sample in original dimensions
    """
    print(probs.shape, gt.shape)
    assert probs.shape[:-1] == gt.shape

    bs, h, w, c = probs.shape
    probs = probs.view(bs, h * w, c)
    gt_indices = (
        gt.reshape((bs, -1, 1)).expand(-1, -1, probs.shape[2]).type(torch.int64)
    )  # (bs, n_pixels, n_classes)
    value_at_ground_truth = torch.gather(probs, 2, gt_indices)[
        :, :, 0
    ]  # (bs, n_pixels)
    value_at_ground_truth = value_at_ground_truth.reshape(bs, h, w)

    return value_at_ground_truth.cpu()  # bs, h, w


def _calculate_self_confidence_threshold(
    probs: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    bs, h, w, c = probs.shape
    value_at_ground_truth = _calculate_self_confidence(probs, gt)
    # count = torch.bincount(gt.view(-1), minlength=probs.shape[-1])

    # get the mean of the self confidence per class
    mean_self_confidence_per_class = []
    for i in range(c):
        mask = gt == i
        mean_self_confidence = torch.mean(value_at_ground_truth[mask])
        mean_self_confidence_per_class.append(mean_self_confidence.item())
    return mean_self_confidence_per_class


def new_get_mislabeled_samples(probs: torch.Tensor, gt: torch.Tensor) -> None:
    _calculate_self_confidence(probs, gt)
    count_per_class = torch.bincount(gt.view(-1), minlength=probs.shape[-1])


def _get_per_class_threshold(df: DataFrame, num_classes: int) -> np.ndarray:
    """Gets the average self-confidence per class

    average self-confidence: The average probabilities at a class for
        samples labeled as that class
    """
    selection = []
    for idx in range(num_classes):
        df.select(f"{VC.gold}=={idx}", name=f"{VC.gold}_{idx}")
        selection.append(f"{VC.gold}_{idx}")
    return df[SELF_CONFIDENCE].mean(selection=selection)


def _get_confidence_joint(df: DataFrame) -> np.ndarray:
    """Calculates and normalizes the confident_counts into a confidence_joint"""
    # This is in format {gold_idx: count}
    class_count_dict = df[VC.gold].value_counts().sort_index().to_dict()
    # If there are any missing classes we want to fill them with 0s
    num_classes = df[VC.prob].shape[1]
    count_per_class = np.array(
        [class_count_dict.get(class_idx, 0) for class_idx in range(num_classes)]
    )
    per_class_threshold = _get_per_class_threshold(df, num_classes)
    confident_counts = _get_confident_counts(df, per_class_threshold, num_classes)
    confidence_joint = normalize_confident_counts(confident_counts, count_per_class)
    return confidence_joint
