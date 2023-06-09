from typing import List

import torch

from dataquality.clients.objectstore import ObjectStore

object_store = ObjectStore()


def calculate_lm_for_batch(
    self_confidence: torch.Tensor,
    confident_count: torch.Tensor,
    class_counts: torch.Tensor,
    gold: torch.Tensor,
    number_classes: int,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Caluclate likely mislabeled for a batch and return in image form

    Args:
        self_confidence (torch.Tensor): confidence of each sample at gold (bs, h, w)
        confident_count (torch.Tensor): matrix of (classes, classes) of counts of
            confident predictions for each class
        class_counts (torch.Tensor): number of samples for each class (classes,)
        gold (torch.Tensor): ground truth labels for each sample (bs, h, w)
        number_of_samples (int): number_clases
        probs (torch.Tensor): probabilities for each sample (bs, h, w, classes)
    Returns:
        torch.Tensor: bs, h, w matrix of 1s and 0s where 1s are likely mislabeled
    """
    confident_joint = normalize_confident_counts(confident_count, class_counts)
    mislabeled_by_conf = get_mislabeled_by_class_confidence(
        self_confidence, confident_joint, gold=gold, num_classes=number_classes
    )
    mislabeled_by_noise = get_mislabeled_by_noise(confident_joint, probs, gold)

    final_mislabeled = mislabeled_by_conf * mislabeled_by_noise

    return final_mislabeled


def get_mislabeled_by_noise(
    confident_joint: torch.Tensor,
    probs: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    """Gets the most likely mislabeled samples per given GT and actual GT pair using

    the confidence_joint. See the CL paper for more information, but using the
    confidence joint we can get an estimated probability for the percentage of
    samples given the i GT label but actually belonging to the j GT label.

    We extrapolate from the probability a number of samples that are most likely
    to be mislabeled for each cell in the CJ matrix and choose that many of the
    highest margin samples
      (highest margin between predicted to be class j but given class i).

    Args:
        confident_joint (torch.Tensor): confidence joint matrix
        probs (torch.Tensor): bs, h, w, classes matrix of probabilities
        gold (torch.Tensor): bs h, w matrix of ground truth labels

    Returns:
        torch.Tensor: bs, h, w matrix of 1s and 0s where 1s are likely mislabeled
    """

    original_shape = gold.shape
    num_classes = probs.shape[-1]
    probs = probs.view(-1, num_classes)
    num_samples = probs.shape[0]
    gold = gold.view(-1)
    mislabled_ids: List = []
    ids = torch.arange(probs.shape[0])
    for given_gold_idx in range(num_classes):
        bool_mask = gold == given_gold_idx
        class_probs = probs[bool_mask]
        sample_ids = ids[bool_mask]
        out = calculate_mislabled_by_noise(
            confident_joint, class_probs, given_gold_idx, num_samples
        )
        mislabled_ids.extend(sample_ids[out])

    mislabled_ids_tensor = torch.tensor(mislabled_ids)
    final_mislabeled = torch.zeros(gold.shape).view(-1)
    final_mislabeled[mislabled_ids_tensor.view(-1).long()] = 1
    final_mislabeled = final_mislabeled.view(original_shape)
    return final_mislabeled


def calculate_mislabled_by_noise(
    confident_joint: torch.Tensor,
    probs: torch.Tensor,
    given_gold_idx: int,
    num_samples: int,
) -> List[int]:
    """Calculated mislabled for a given gold index

    Args:
        confident_joint (torch.Tensor): confidence joint matrix
        probs (torch.Tensor): matrix of probabilities for each sample
            where the gold label is given_gold_idx
        given_gold_idx (torch.Tensor): the gold label we are looking at
        num_samples (int): number of samples in probs

    Returns:
        List[int]: list of sample ids that are most likely mislabeled
    """

    mislabeled_idxs_in_class: List[int] = []
    num_classes = probs.shape[-1]
    probs = probs.view(-1, num_classes)
    # "actual" or "expected" gold index, looking at each class one by one
    for actual_gold_idx in range(num_classes):  # j in the paper
        if given_gold_idx == actual_gold_idx:
            continue

        # number of mislabeled samples for this given / actual GT pair
        num_mislabeled = int(
            torch.round(
                confident_joint[given_gold_idx, actual_gold_idx] * num_samples
            ).item()
        )
        if not num_mislabeled:
            continue

        # Margin is how much more confident the model is in a non-given GT label
        class_samples_margin = probs[:, actual_gold_idx] - probs[:, given_gold_idx]
        # We want the top `num_mislabeled` samples with the largest margin
        # for the actual/expected class relative to the given class
        num_mislabeled = min(num_mislabeled, class_samples_margin.shape[0])
        _, topk_idxs = (-class_samples_margin).topk(num_mislabeled)
        mislabeled_idxs_in_class.extend(topk_idxs.tolist())
    return mislabeled_idxs_in_class


def get_mislabeled_by_class_confidence(
    self_confidence: torch.Tensor,
    confidence_joint: torch.Tensor,
    gold: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Gets the most likely mislabeled samples per class using the confidence_joint

    See the CL paper for more information, but using the confidence joint we can get
    an estimated probability for the percentage of a class that's likely mislabeled.
    We extrapolate from the probaiblity a number of samples that are most likely to
    be mislabeled for each class and choose that many of the lowest self-confidence
    samples.

    (1) get self confidence for all samples, (2) sort ascending across all, (3) get
    number_of_samples_in_this_class_likely_mislabeled per class and (4) take the top
    number_of_samples_in_this_class_likely_mislabeled per class with the lowest
    self confidence

    Args:
        self_confidence: self confidence for all samples (bs, h, w)
        confidence_joint: confidence joint for all samples (num_classes, num_classes)
        gold: ground truth labels for all samples (bs, h, w)

    Returns:
        torch.Tensor: mislabeled samples (bs, h, w)

    """
    original_shape = gold.shape
    ids = torch.arange(self_confidence.view(-1).shape[0])
    self_confidence = self_confidence.view(-1)
    gold = gold.view(-1)
    num_samples = self_confidence.shape[0]

    # stable sort by self confidence with ids following along
    _, sorted_indices = torch.sort(self_confidence)
    sorted_ids = ids[sorted_indices]
    sorted_gold = gold[sorted_indices]
    num_mislabeled_per_class = torch.round(
        (confidence_joint.sum(dim=1) - torch.diag(confidence_joint)) * num_samples
    ).to(torch.int)

    mislabeled_ids = []
    for class_idx, num_mislabeled in zip(range(num_classes), num_mislabeled_per_class):
        bool_mask = sorted_gold == class_idx
        bool_ids = sorted_ids[bool_mask]
        mislabeled_ids += bool_ids[:num_mislabeled].tolist()

    mislabeled_ids_tensor = torch.tensor(mislabeled_ids).to(torch.int)
    final_mislabeled = torch.zeros(gold.shape).view(-1)
    final_mislabeled[mislabeled_ids_tensor.view(-1).long()] = 1
    final_mislabeled = final_mislabeled.view(original_shape)

    return final_mislabeled


def calculate_self_confidence(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Gets the self confidence for each sample meaning the
    confidence in a prediction of its given GT label

    Args:
        probs (torch.Tensor): probability mask for each sample
        gold (torch.Tensor): ground truth label for each sample

    Returns:
        torch.Tensor: self confidence for each sample in original dimensions
    """
    assert probs.shape[:-1] == gold.shape

    bs, h, w, c = probs.shape
    probs = probs.view(bs, h * w, c)
    gold_indices = (
        gold.reshape((bs, -1, 1)).expand(-1, -1, probs.shape[2]).type(torch.int64)
    )  # (bs, n_pixels, n_classes)
    value_at_gold = torch.gather(probs, 2, gold_indices)[:, :, 0]  # (bs, n_pixels)
    value_at_gold = value_at_gold.reshape(bs, h, w)

    return value_at_gold.cpu()  # bs, h, w


def fill_confident_counts(
    probs: torch.Tensor,
    gold: torch.Tensor,
    given_class: int,
    per_class_threshold: torch.Tensor,
    confident_counts: torch.Tensor,
) -> torch.Tensor:
    """Helper function that fills the confident counts
       based on how many values are above the threshold
       for a given class and are labeled a given class

       ex. if label is a but above threshold for b
       increment confident_counts[a, b] by 1

       Read the confidence count as: of all of those with above
       the threshold confidence in class a this is how many were labeled
       class b

    Args:
        probs (torch.Tensor): probability mask for a give class
        gold (torch.Tensor): label mask for each sample
        given_class (int): the class we are looking at
        per_class_threshold (np.ndarray): per class threshold
        confident_counts (np.ndarray): matrix of confident counts

    Returns:
        np.ndarray: confident counts matrix
    """
    boolean_mask = probs >= per_class_threshold
    # if it is above the threshold find the label and increment the count
    labels = gold[boolean_mask]
    # increment the count for each label
    # get the count of each label
    count_labels = torch.bincount(labels, minlength=confident_counts.shape[1])
    confident_counts[given_class, :] = count_labels.cpu()
    return confident_counts


def normalize_confident_counts(
    confident_counts: torch.Tensor, class_counts: torch.Tensor
) -> torch.Tensor:
    """Normalize the confident_counts -> confidence joint.

    Going from a count matrix to a valid probability matrix by making sure that the rows
    sum to the class distribution. In case this batch doesnt have all classes
    represented, fill with 0

    class_counts is the count of number of samples per unique class in the data
    """
    # If a row got no samples, add a 1 on their diagonal (pretend there is no noise).
    labels_with_no_samples = torch.where(confident_counts.sum(dim=1) == 0.0)[0]
    if len(labels_with_no_samples) > 0:
        partial_diag = torch.zeros(
            (len(class_counts), len(class_counts)), dtype=torch.int64
        )
        partial_diag[labels_with_no_samples, labels_with_no_samples] = 1
        confident_counts += partial_diag

    # Normalize each row so that they represent the distribution of labels.
    confidence_joint = (
        confident_counts
        * torch.unsqueeze(class_counts, dim=1)
        / (confident_counts.sum(dim=1, keepdim=True) + 1e-10)
    )
    # Normalize the entire matrix to have a distribution.
    confidence_joint /= confidence_joint.sum() + 1e-10

    # This accounts for the unlikely scenario that class counts is 0.
    if torch.allclose(confidence_joint.sum(), torch.tensor(0.0)):
        confidence_joint.fill_diagonal_(1.0 / confidence_joint.shape[0])

    return confidence_joint


def calculate_self_confidence_threshold(
    probs: torch.Tensor, gold: torch.Tensor
) -> List[float]:
    """Calculates the self confidence threshold for each class

    Args:
        probs (torch.Tensor): bs, h, w, c probability mask
        gold (torch.Tensor): bs, h, w label mask

    Returns:
        torch.Tensor: (classes, ) self confidence threshold for each class
    """
    bs, h, w, c = probs.shape
    value_at_gold = calculate_self_confidence(probs, gold)

    # get the mean of the self confidence per class
    mean_self_confidence_per_class = []
    for i in range(c):
        mask = gold == i
        mean_self_confidence = torch.mean(value_at_gold[mask])
        mean_self_confidence_per_class.append(mean_self_confidence.item())
    return mean_self_confidence_per_class
