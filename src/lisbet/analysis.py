"""LISBET analysis.
Module for analyzing sequences of states (motifs) in a dataset.

This module provides functions to compute various statistics for sequences of states,
including bout statistics, transition probabilities, and F1 score matrices. The
functions are designed to work with sequences of states, where each state represents a
motif in the sequence.

Functions
---------
bout_stats(sequences, lengths, fps, groups=None)
    Compute statistics for sequences of states (motifs) in a dataset.

transition_probability(sequences, lengths, dummy_state_id=None, groups=None)
    Compute transition probabilities between states in a sequence.

f1_score_matrix(labels, predictions)
    Compute an F1 score matrix comparing predicted states to true labels.

gwet_ac1_score(labels, predictions, classes=None)
    Compute unweighted Gwet's AC1 for nominal multicategory ratings.

frame_agreement_metrics(labels, predictions, classes=None)
    Compute weighted F1, balanced accuracy, Cohen's kappa, and Gwet's AC1.

matched_bout_iou(labels, predictions, classes, iou_threshold=0.5)
    Compute macro-averaged IoU across one-to-one matched bouts.

behavior_agreement_metrics(...)
    Compute the complete frame-wise and matched-bout metric set.

"""

from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    recall_score,
)
from tqdm.auto import trange


def _paired_1d_arrays(labels, predictions):
    """Validate and return paired one-dimensional label arrays."""
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if labels.shape != predictions.shape:
        raise ValueError(
            "labels and predictions must contain the same number of elements."
        )
    if labels.size == 0:
        raise ValueError("labels and predictions must not be empty.")
    if pd.isna(labels).any() or pd.isna(predictions).any():
        raise ValueError("labels and predictions must not contain missing values.")

    return labels, predictions


def _resolve_classes(labels, predictions, classes=None):
    """Resolve and validate the category set used by agreement metrics."""
    observed = np.unique(np.concatenate([labels, predictions]))
    if classes is None:
        return observed

    classes = np.asarray(classes).reshape(-1)
    if classes.size == 0:
        raise ValueError("classes must contain at least one category.")
    if not np.isin(observed, classes).all():
        raise ValueError("classes must include every observed label and prediction.")

    return classes


def bout_stats(sequences, lengths, fps, groups=None):
    """
    Compute statistics for sequences of states (motifs) in a dataset.

    This function calculates the mean bout duration and event rate for each motif in
    the sequence.

    Parameters
    ----------
    sequences : list or array-like
        The sequence of states, where each state represents a motif.
    lengths : list or array-like
        The length (duration) of each sequence. Must be the same length as `sequences`.
    fps : int or float
        The frame rate at which the sequences were recorded. Used to convert bout
        durations and event rates to seconds and minutes respectively.
    groups : list or array-like, optional
        A list of group labels corresponding to each sequence. If None, all sequences
        are assumed to belong to a default group. Default is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns: "Motif ID", "Group label", "Mean bout
        duration (s)", "Rate (events / min)". Each row corresponds to a unique motif
        in the sequences, and the DataFrame is grouped by the group label if provided.

    """
    analysis_results = []
    for i, seq_duration in enumerate(lengths):
        # Select sequence data
        start = sum(lengths[:i])
        stop = start + seq_duration
        seq_pred = sequences[start:stop]

        # Set group label, if available
        group_label = groups[i] if groups is not None else "default"

        # Compute sequence of states
        events = pd.DataFrame(
            [(k, sum(1 for i in g)) for k, g in groupby(seq_pred)],
            columns=["motif_id", "bout_duration"],
        )

        # Compute statistics
        events_stats = events.groupby("motif_id").agg(["mean", "std", "count", "sum"])

        for motif_id in events_stats.index:
            analysis_results.append(
                {
                    "Motif ID": motif_id,
                    "Group label": group_label,
                    "Mean bout duration (s)": (
                        events_stats.loc[motif_id, "bout_duration"]["mean"] / fps
                    ),
                    "Rate (events / min)": events_stats.loc[motif_id, "bout_duration"][
                        "count"
                    ]
                    / (events_stats["bout_duration"]["sum"].sum() / fps / 60),
                }
            )

    analysis_results = pd.DataFrame(
        analysis_results,
        columns=[
            "Motif ID",
            "Group label",
            "Mean bout duration (s)",
            "Rate (events / min)",
        ],
    )

    return analysis_results


def transition_probability(sequences, lengths, dummy_state_id=None, groups=None):
    """
    Compute transition probabilities between states in a sequence.

    This function calculates the probability of transitioning from one state to another
    in a sequence of states, optionally ignoring a specified dummy state.

    Parameters
    ----------
    sequences : list or array-like
        The sequence of states, where each state represents a motif.
    lengths : list or array-like
        The length (duration) of each sequence. Must be the same length as `sequences`.
    dummy_state_id : int, optional
        The state ID to be ignored in the analysis (e.g., a dummy state). If None, no
        state is ignored. Default is None.
    groups : list or array-like, optional
        A list of group labels corresponding to each sequence. If None, all sequences
        are assumed to belong to a default group. Default is None.

    Returns
    -------
    dict
        A dictionary where each key is a group label and each value is a 2D numpy array
        representing the transition probability matrix for that group. The matrix has
        dimensions (number of states) x (number of states).

    """
    # Find number of states
    assert np.min(sequences) == 0
    num_states = int(np.max(sequences)) + 1

    if dummy_state_id is not None:
        num_states = num_states - 1

    # Count transitions
    if groups is None:
        occurrences = {"default": np.zeros((num_states, num_states))}
        groups = ["default"] * len(lengths)
    else:
        occurrences = {k: np.zeros((num_states, num_states)) for k in np.unique(groups)}

    for j, seq_duration in enumerate(lengths):
        # Select sequence data
        start = sum(lengths[:j])
        stop = start + seq_duration
        seq_pred = sequences[start:stop]
        group = groups[j]

        # Compute sequence of states, ignoring the dummy state
        events = [
            (k, sum(1 for i in g)) for k, g in groupby(seq_pred) if k != dummy_state_id
        ]

        for i in range(len(events) - 1):
            src = events[i][0]
            dst = events[i + 1][0]

            # Skip self-events, introduced by ignoring the dummy state
            if src != dst:
                occurrences[group][src][dst] += 1

    # Make probability
    trans_prob = {
        group: occurrences[group] / np.sum(occurrences[group], axis=1)[:, np.newaxis]
        for group in occurrences
    }

    return trans_prob


def f1_score_matrix(labels, predictions):
    """
    Compute an F1 score matrix comparing predicted states to true labels.

    This function calculates the F1 score for each pair of predicted and true states,
    resulting in a matrix of F1 scores where each row corresponds to a predicted state
    and each column corresponds to a true state.

    Parameters
    ----------
    labels : list or array-like
        The true state labels.
    predictions : list or array-like
        The predicted state labels.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array where each element [i, j] represents the F1 score for
        predicting state i as state j.

    """
    n_states = np.max(predictions) + 1
    n_classes = np.max(labels) + 1

    f1_matrix = []
    for s in trange(n_states, desc="Computing F1 score"):
        bin_pred = np.array(predictions == s, dtype=int)
        score = []
        for lbl in range(n_classes):
            bin_lab = np.array(labels == lbl, dtype=int)
            score.append(f1_score(bin_lab, bin_pred, zero_division=0.0))
        f1_matrix.append(score)
    f1_matrix = np.array(f1_matrix)

    return f1_matrix


def gwet_ac1_score(labels, predictions, classes=None):
    """Compute unweighted Gwet's AC1 for nominal multicategory ratings.

    Parameters
    ----------
    labels : list or array-like
        Reference frame labels.
    predictions : list or array-like
        Predicted frame labels.
    classes : list or array-like, optional
        Complete category set. Supplying this argument is recommended when a fixed
        category vocabulary must be used even when one category is absent from a
        particular recording.

    Returns
    -------
    float
        Gwet's AC1 agreement coefficient.

    Notes
    -----
    This is the unweighted, two-rater, nominal multicategory AC1. The expected
    agreement is ``sum_k p_k * (1 - p_k) / (q - 1)``, where ``p_k`` is the pooled
    marginal proportion for category ``k`` and ``q`` is the number of categories.
    """
    labels, predictions = _paired_1d_arrays(labels, predictions)
    classes = _resolve_classes(labels, predictions, classes)

    observed_agreement = float(np.mean(labels == predictions))
    n_categories = len(classes)
    if n_categories == 1:
        return observed_agreement

    pooled_proportions = np.asarray(
        [
            (np.sum(labels == category) + np.sum(predictions == category))
            / (2 * labels.size)
            for category in classes
        ],
        dtype=float,
    )
    chance_agreement = float(
        np.sum(pooled_proportions * (1 - pooled_proportions))
        / (n_categories - 1)
    )
    denominator = 1 - chance_agreement

    if np.isclose(denominator, 0):
        return 1.0 if np.isclose(observed_agreement, 1) else np.nan

    return float((observed_agreement - chance_agreement) / denominator)


def frame_agreement_metrics(labels, predictions, classes=None):
    """Compute the four frame-wise agreement metrics used in the evaluation.

    Parameters
    ----------
    labels : list or array-like
        Reference frame labels.
    predictions : list or array-like
        Predicted frame labels.
    classes : list or array-like, optional
        Complete category set. For CalMS21 Task 2, pass
        ``["attack", "investigation", "mount", "other"]``.

    Returns
    -------
    dict
        Dictionary containing ``weighted_f1``, ``balanced_accuracy``,
        ``cohen_kappa``, and ``gwet_ac1``.

    Notes
    -----
    Balanced accuracy is computed as the unweighted mean recall over ``classes``.
    When a fixed category set is supplied, a class absent from the reference labels
    receives recall zero for that recording.
    """
    labels, predictions = _paired_1d_arrays(labels, predictions)
    classes = _resolve_classes(labels, predictions, classes)

    return {
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=classes,
                average="weighted",
                zero_division=0.0,
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                labels,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0.0,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(labels, predictions, labels=classes)
        ),
        "gwet_ac1": gwet_ac1_score(
            labels,
            predictions,
            classes=classes,
        ),
    }


def _class_bouts(sequence, class_label):
    """Return half-open ``(start, stop)`` bouts for one class."""
    sequence = np.asarray(sequence).reshape(-1)
    mask = sequence == class_label
    if not np.any(mask):
        return []

    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _bout_iou(reference_bout, predicted_bout):
    """Compute temporal IoU between two half-open bouts."""
    reference_start, reference_stop = reference_bout
    predicted_start, predicted_stop = predicted_bout

    intersection = max(
        0,
        min(reference_stop, predicted_stop)
        - max(reference_start, predicted_start),
    )
    union = max(reference_stop, predicted_stop) - min(
        reference_start, predicted_start
    )
    return intersection / union if union > 0 else 0.0


def matched_bout_iou(
    labels,
    predictions,
    classes,
    iou_threshold=0.5,
):
    """Compute macro-averaged IoU across greedily matched bouts.

    Parameters
    ----------
    labels : list or array-like
        Reference frame labels.
    predictions : list or array-like
        Predicted frame labels.
    classes : list or array-like
        Classes included in the bout-wise macro-average. For CalMS21 Task 2, pass
        ``["attack", "investigation", "mount"]`` to exclude ``"other"``.
    iou_threshold : float, optional
        Minimum temporal IoU required for a matched pair. Default is 0.5.

    Returns
    -------
    float
        Macro-average across classes of the mean IoU of successfully matched bouts.
        Returns ``numpy.nan`` when no class contains a successful match.

    Notes
    -----
    This reproduces the manuscript analysis. Within each class, all candidate
    reference-prediction bout pairs meeting ``iou_threshold`` are ranked from
    highest to lowest IoU and greedily matched one-to-one. Each reference and
    predicted bout can be used at most once. The metric is conditional on successful
    matching and therefore measures temporal localization, not bout-detection
    precision or recall. Classes with no successful match are omitted from the
    macro-average.
    """
    labels, predictions = _paired_1d_arrays(labels, predictions)
    classes = np.asarray(classes).reshape(-1)

    if classes.size == 0:
        raise ValueError("classes must contain at least one category.")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1.")

    class_scores = []
    for class_label in classes:
        reference_bouts = _class_bouts(labels, class_label)
        predicted_bouts = _class_bouts(predictions, class_label)

        candidates = []
        for reference_index, reference_bout in enumerate(reference_bouts):
            for prediction_index, predicted_bout in enumerate(predicted_bouts):
                iou = _bout_iou(reference_bout, predicted_bout)
                if iou >= iou_threshold:
                    candidates.append((iou, reference_index, prediction_index))

        matched_reference = set()
        matched_prediction = set()
        matched_ious = []

        # Descending tuple sort reproduces the original Task 2 implementation.
        for iou, reference_index, prediction_index in sorted(
            candidates, reverse=True
        ):
            if reference_index in matched_reference:
                continue
            if prediction_index in matched_prediction:
                continue

            matched_reference.add(reference_index)
            matched_prediction.add(prediction_index)
            matched_ious.append(iou)

        if matched_ious:
            class_scores.append(float(np.mean(matched_ious)))

    return float(np.mean(class_scores)) if class_scores else np.nan


def behavior_agreement_metrics(
    labels,
    predictions,
    frame_classes=None,
    bout_classes=None,
    iou_threshold=0.5,
):
    """Compute the complete frame-wise and matched-bout evaluation set.

    Parameters
    ----------
    labels : list or array-like
        Reference frame labels.
    predictions : list or array-like
        Predicted frame labels.
    frame_classes : list or array-like, optional
        Classes used for frame-wise metrics.
    bout_classes : list or array-like, optional
        Classes used for matched-bout IoU. If None, the bout-wise metric is omitted.
    iou_threshold : float, optional
        Minimum temporal IoU for a matched bout pair. Default is 0.5.

    Returns
    -------
    dict
        Frame-wise metrics and, when requested, ``matched_bout_iou``.
    """
    metrics = frame_agreement_metrics(
        labels,
        predictions,
        classes=frame_classes,
    )

    if bout_classes is not None:
        metrics["matched_bout_iou"] = matched_bout_iou(
            labels,
            predictions,
            classes=bout_classes,
            iou_threshold=iou_threshold,
        )

    return metrics
