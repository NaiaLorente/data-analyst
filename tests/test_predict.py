"""Unit tests for the 'What predicts this?' feature-importance module."""

import base64

import numpy as np
import pandas as pd
from agent.predict import predict_importance, plot_feature_importance


def _classification_df(n=200):
    rng = np.random.RandomState(42)
    strong = rng.normal(size=n)
    noise = rng.normal(size=n)
    # target is (mostly) determined by `strong`, not `noise` — a real signal to detect
    target = (strong + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "strong_signal": strong,
            "pure_noise": noise,
            "category": rng.choice(["A", "B", "C"], size=n),
            "target": target,
        }
    )


def _regression_df(n=200):
    rng = np.random.RandomState(42)
    strong = rng.normal(size=n)
    noise = rng.normal(size=n)
    target = strong * 10 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"strong_signal": strong, "pure_noise": noise, "target": target})


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_importance_classification_detects_binary_int_target():
    df = _classification_df()
    result = predict_importance(df, "target")
    assert result["applicable"] is True
    assert result["task"] == "classification"
    assert result["score_name"] == "accuracy"
    assert result["score"] > 0.6  # should clearly beat a coin flip


def test_predict_importance_classification_ranks_real_signal_above_noise():
    df = _classification_df()
    result = predict_importance(df, "target")
    top_feature = result["features"][0]["column"]
    assert top_feature == "strong_signal"


def test_predict_importance_regression_task():
    df = _regression_df()
    result = predict_importance(df, "target")
    assert result["applicable"] is True
    assert result["task"] == "regression"
    assert result["score_name"] == "r2"
    assert result["score"] > 0.5


def test_predict_importance_handles_string_target_as_classification():
    df = _classification_df()
    df["label"] = df["target"].map({0: "no", 1: "yes"})
    result = predict_importance(df.drop(columns=["target"]), "label")
    assert result["task"] == "classification"


def test_predict_importance_not_applicable_with_too_few_rows():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    result = predict_importance(df, "target")
    assert result["applicable"] is False
    assert "reason" in result


def test_predict_importance_not_applicable_with_single_class_target():
    df = pd.DataFrame({"a": range(30), "target": [1] * 30})
    result = predict_importance(df, "target")
    assert result["applicable"] is False


def test_predict_importance_not_applicable_with_no_feature_columns():
    df = pd.DataFrame({"target": range(30)})
    result = predict_importance(df, "target")
    assert result["applicable"] is False


def test_predict_importance_handles_missing_values_in_features():
    df = _classification_df()
    df.loc[0:10, "strong_signal"] = None
    result = predict_importance(df, "target")
    assert result["applicable"] is True


def test_plot_feature_importance_produces_valid_png():
    features = [{"column": "a", "importance": 0.5}, {"column": "b", "importance": 0.2}]
    assert _is_valid_png(plot_feature_importance(features))
