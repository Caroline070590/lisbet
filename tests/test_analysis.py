"""Tests for the analysis module."""

import numpy as np
import pandas as pd
import pytest

from lisbet import analysis


class TestBoutStats:
    def test_bout_stats_basic(self):
        sequences = [1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 1, 1, 2]
        lengths = [8, 5]
        fps = 30

        result = analysis.bout_stats(sequences, lengths, fps)

        expected_data = {
            "Motif ID": [1, 2, 1, 2, 3],
            "Group label": ["default"] * 5,
            "Mean bout duration (s)": [
                (2 + 3) / (2 * fps),
                3 / fps,
                2 / fps,
                1 / fps,
                2 / fps,
            ],
            "Rate (events / min)": [
                (2 * 60 * fps) / 8,  # 2 events : 8 frames = n events : 60*FPS frames
                (1 * 60 * fps) / 8,
                (1 * 60 * fps) / 5,
                (1 * 60 * fps) / 5,
                (1 * 60 * fps) / 5,
            ],
        }

        expected_result = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result, expected_result)


class TestAgreementMetrics:
    def test_gwet_ac1_score(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 1])

        result = analysis.gwet_ac1_score(
            labels,
            predictions,
            classes=[0, 1],
        )

        assert result == pytest.approx(0.5294117647058824)

    def test_frame_agreement_metrics(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 1])

        result = analysis.frame_agreement_metrics(
            labels,
            predictions,
            classes=[0, 1],
        )

        assert result == pytest.approx(
            {
                "weighted_f1": 0.7333333333333334,
                "balanced_accuracy": 0.75,
                "cohen_kappa": 0.5,
                "gwet_ac1": 0.5294117647058824,
            }
        )

    def test_matched_bout_iou(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
        predictions = np.array([0, 0, 1, 1, 0, 0, 2, 2, 2])

        result = analysis.matched_bout_iou(
            labels,
            predictions,
            classes=[1, 2],
            iou_threshold=0.5,
        )

        assert result == pytest.approx(2 / 3)

    def test_matched_bout_iou_uses_greedy_matching(self, monkeypatch):
        labels = np.array([1, 1, 0, 1, 1])
        predictions = np.array([1, 0, 1, 1, 0])

        score_by_pair = {
            ((0, 2), (0, 1)): 0.9,
            ((0, 2), (2, 4)): 0.8,
            ((3, 5), (0, 1)): 0.8,
            ((3, 5), (2, 4)): 0.0,
        }

        monkeypatch.setattr(
            analysis,
            "_bout_iou",
            lambda reference, prediction: score_by_pair[(reference, prediction)],
        )

        result = analysis.matched_bout_iou(
            labels,
            predictions,
            classes=[1],
            iou_threshold=0.5,
        )

        # Greedy matching selects the highest-IoU pair (0.9) first, which blocks
        # the two 0.8 alternatives. This differs from a Hungarian assignment.
        assert result == pytest.approx(0.9)

    def test_behavior_agreement_metrics(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
        predictions = np.array([0, 0, 1, 1, 0, 0, 2, 2, 2])

        result = analysis.behavior_agreement_metrics(
            labels,
            predictions,
            frame_classes=[0, 1, 2],
            bout_classes=[1, 2],
        )

        assert set(result) == {
            "weighted_f1",
            "balanced_accuracy",
            "cohen_kappa",
            "gwet_ac1",
            "matched_bout_iou",
        }
        assert result["matched_bout_iou"] == pytest.approx(2 / 3)
