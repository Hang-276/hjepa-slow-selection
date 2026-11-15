"""Finite-window physical scores and fixed-threshold selective error."""

import numpy as np
from scipy.stats import beta


def _variance(values):
    variance = np.asarray(values, dtype=np.float64)
    if variance.shape != (4,) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Reference variances must contain four finite, positive values")
    return variance


def _top2_and_gap(scores):
    order = np.argsort(scores, axis=1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=1)
    return np.sort(order[:, :2], axis=1), ordered[:, 2] - ordered[:, 1]


def score_windows(prediction, true_states, physical_ref_variance, predicted_ref_variance, lags):
    """Score matched [window, time, 4] paths in player-x/y, distractor-0-x/y order.

    Each lag gets equal weight after averaging its available endpoints. The
    cross term and error energy are unnormalized numerator contributions from
    the error in physical units. Zero-based axis order breaks score ties.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(true_states, dtype=np.float64)
    if (truth.ndim != 3 or truth.shape[-1] != 4 or truth.shape != prediction.shape
            or len(truth) == 0 or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError("Expected aligned, finite [window, time, 4] paths")
    physical_variance = _variance(physical_ref_variance)
    predicted_variance = _variance(predicted_ref_variance)
    lags = tuple(lags)
    if not lags or any(not isinstance(h, (int, np.integer)) or isinstance(h, (bool, np.bool_))
                       or not 0 < h < truth.shape[1] for h in lags):
        raise ValueError("Scoring lags must be positive integers shorter than the window")

    true_num, predicted_num, cross_term, error_energy = np.zeros((4, len(truth), 4))
    for h in lags:
        true_delta = truth[:, h:] - truth[:, :-h]
        predicted_delta = prediction[:, h:] - prediction[:, :-h]
        error_delta = predicted_delta - true_delta
        true_num += np.mean(true_delta**2, axis=1)
        predicted_num += np.mean(predicted_delta**2, axis=1)
        cross_term += 2 * np.mean(true_delta * error_delta, axis=1)
        error_energy += np.mean(error_delta**2, axis=1)
    true_scores = (true_num / len(lags)) / physical_variance
    predicted_scores = (predicted_num / len(lags)) / predicted_variance
    true_top2, true_gap = _top2_and_gap(true_scores)
    predicted_top2, gap = _top2_and_gap(predicted_scores)
    return {
        "true_scores": true_scores,
        "predicted_scores": predicted_scores,
        "true_top2": true_top2,
        "predicted_top2": predicted_top2,
        "gap": gap,
        "wrong": (true_top2 != predicted_top2).any(axis=1),
        "true_gap": true_gap,
        "cross_term": cross_term / len(lags),
        "error_energy": error_energy / len(lags),
    }


def risk(gap, wrong, threshold, alpha, minimum_coverage=0.35, max_error=0.01):
    """Report selective error and its one-sided Clopper–Pearson upper bound.

    Use the caller's per-claim alpha and a threshold fixed before evaluation.
    The binomial bound assumes independent windows within the evaluated cell.
    """
    gap = np.asarray(gap, dtype=np.float64)
    wrong = np.asarray(wrong)
    if (gap.ndim != 1 or wrong.shape != gap.shape or len(gap) == 0
            or not np.isfinite(gap).all() or not np.isin(wrong, (False, True)).all()):
        raise ValueError("Expected a nonempty finite gap vector and a matching wrong indicator")
    if not 0 < alpha < 1 or not 0 <= minimum_coverage <= 1 or not 0 <= max_error <= 1 or np.isnan(threshold):
        raise ValueError("Invalid threshold, tail probability, or pass criteria")
    accept = gap >= threshold
    accepted = int(accept.sum())
    errors = int(np.count_nonzero(wrong[accept]))
    coverage = accepted / len(gap)
    upper = 1.0 if errors == accepted else float(beta.ppf(1 - alpha, errors + 1, accepted - errors))
    coverage_pass = coverage >= minimum_coverage
    error_pass = accepted > 0 and upper <= max_error
    return {
        "windows": len(gap),
        "accepted": accepted,
        "errors": errors,
        "coverage": coverage,
        "selective_error": errors / accepted if accepted else None,
        "error_upper": upper,
        "passes_coverage": coverage_pass,
        "passes_error_bound": error_pass,
        "passes": coverage_pass and error_pass,
    }
