import numpy as np
import pytest

from probly.evaluation.ood_api import (
    evaluate_ood,
    parse_dynamic_metric,
    STATIC_METRICS,
    DYNAMIC_METRICS,
)



# Fixtures


@pytest.fixture
def perfectly_separated_scores():
    """Perfect separation: ID scores always higher than OOD scores."""
    in_scores = np.array([0.9, 0.8, 0.7, 0.6])
    out_scores = np.array([0.4, 0.3, 0.2, 0.1])
    return in_scores, out_scores


@pytest.fixture
def identical_scores():
    """Identical distributions: no separation possible."""
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    return scores, scores



# parse_dynamic_metric – valid specifications


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("fpr", ("fpr", 0.95)),
        ("fnr", ("fnr", 0.95)),
        ("fpr@0.8", ("fpr", 0.8)),
        ("fnr@95%", ("fnr", 0.95)),
        ("FPR@80%", ("fpr", 0.8)),
        (" fnr @ 0.9 ", ("fnr", 0.9)),
    ],
)
def test_parse_dynamic_metric_valid(spec, expected):
    assert parse_dynamic_metric(spec) == expected



# parse_dynamic_metric – invalid specifications


@pytest.mark.parametrize(
    "spec",
    [
        "foo",
        "fpr@",
        "fpr@abc",
        "fpr@-0.1",
        "fpr@0",
        "fpr@1.1",
        "fpr@200%",
    ],
)
def test_parse_dynamic_metric_invalid(spec):
    with pytest.raises(ValueError):
        parse_dynamic_metric(spec)



# evaluate_ood – backward compatibility & return types


def test_default_returns_single_float(perfectly_separated_scores):
    in_s, out_s = perfectly_separated_scores
    result = evaluate_ood(in_s, out_s)
    assert isinstance(result, float)


def test_auroc_string_returns_single_float(perfectly_separated_scores):
    in_s, out_s = perfectly_separated_scores
    result = evaluate_ood(in_s, out_s, metrics="auroc")
    assert isinstance(result, float)



# evaluate_ood – static metrics


@pytest.mark.parametrize("metric", STATIC_METRICS.keys())
def test_static_metrics_return_float(perfectly_separated_scores, metric):
    in_s, out_s = perfectly_separated_scores
    result = evaluate_ood(in_s, out_s, metrics=[metric])

    assert isinstance(result, dict)
    assert metric in result
    assert isinstance(result[metric], float)



# evaluate_ood – dynamic metrics


@pytest.mark.parametrize(
    "metric",
    ["fpr", "fnr", "fpr@0.9", "fnr@90%"],
)
def test_dynamic_metrics_return_float(perfectly_separated_scores, metric):
    in_s, out_s = perfectly_separated_scores
    result = evaluate_ood(in_s, out_s, metrics=[metric])

    assert isinstance(result, dict)
    assert metric in result
    assert isinstance(result[metric], float)


@pytest.mark.parametrize("metric", ["fpr", "fnr", "fpr@0.8"])
def test_dynamic_metrics_are_in_unit_interval(perfectly_separated_scores, metric):
    in_s, out_s = perfectly_separated_scores
    value = evaluate_ood(in_s, out_s, metrics=[metric])[metric]

    assert 0.0 <= value <= 1.0



# evaluate_ood – mathematical edge cases


def test_perfect_separation_auroc_is_one():
    in_scores = np.array([0.1, 0.2, 0.3, 0.4])   # niedrig = ID
    out_scores = np.array([0.6, 0.7, 0.8, 0.9])  # hoch = OOD

    value = evaluate_ood(in_scores, out_scores, metrics="auroc")

    assert value == pytest.approx(1.0)



def test_identical_distributions_auroc_is_half(identical_scores):
    in_s, out_s = identical_scores
    value = evaluate_ood(in_s, out_s, metrics="auroc")

    assert value == pytest.approx(0.5, abs=1e-6)



# evaluate_ood – multiple & all metrics


def test_multiple_metrics_keys_preserved(perfectly_separated_scores):
    in_s, out_s = perfectly_separated_scores
    metrics = ["AUROC", "fpr@0.95"]

    result = evaluate_ood(in_s, out_s, metrics=metrics)

    assert list(result.keys()) == metrics


def test_all_metrics_contains_expected_keys(perfectly_separated_scores):
    in_s, out_s = perfectly_separated_scores
    result = evaluate_ood(in_s, out_s, metrics="all")

    expected = set(STATIC_METRICS.keys()) | {"fpr", "fnr"}
    assert set(result.keys()) == expected



# evaluate_ood – input flexibility


def test_accepts_python_lists():
    result = evaluate_ood(
        in_distribution=[0.9, 0.8, 0.7],
        out_distribution=[0.1, 0.2, 0.3],
        metrics="auroc",
    )

    assert isinstance(result, float)



# evaluate_ood – error handling


def test_unknown_metric_raises_value_error():
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_ood([0.9], [0.1], metrics=["does_not_exist"])


def test_invalid_dynamic_metric_raises_value_error():
    with pytest.raises(ValueError):
        evaluate_ood([0.9], [0.1], metrics=["fpr@200%"])
