""""What predicts this?" — quick baseline feature-importance modeling.

Everything else in this app is descriptive (Auto-Insights, column profiling) or
hypothesis-testing (agent.stats) — this is the first genuinely predictive
capability. Given a target column, trains a small Random Forest (classifier or
regressor, auto-detected from the target's dtype) on the remaining columns and
reports which features actually drive it, ranked by importance. Zero AI cost —
scikit-learn only, no LLM involved, and the AI is never asked to invent or
adjust these numbers.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from agent.charts import CATEGORICAL, INK_SECONDARY, fig_to_base64_png, new_figure

MIN_ROWS = 20


def _is_classification_target(series: pd.Series) -> bool:
    """A numeric column with few distinct whole-number values (0/1 flags, small
    integer codes like a 1-5 rating) is almost always an encoded category, not a
    continuous quantity — e.g. a "Survived" or "Churn" column stored as 0/1 ints,
    extremely common in real datasets. Only genuinely continuous numeric columns
    (many distinct values, or non-integer values) are treated as regression."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        looks_categorical = non_null.nunique() <= 10 and (non_null % 1 == 0).all()
        return bool(looks_categorical)
    return True


def _encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = X.copy()
    encoders = {}
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder
    numeric_medians = X.median(numeric_only=True)
    return X.fillna(numeric_medians).fillna(0), encoders


def predict_importance(df: pd.DataFrame, target: str, max_features: int = 10) -> dict:
    """Train a baseline Random Forest to explain `target` from every other column.
    Returns {"applicable": False, "reason": ...} if there isn't enough usable data."""
    working = df.dropna(subset=[target])
    if len(working) < MIN_ROWS:
        return {"applicable": False, "reason": f"Need at least {MIN_ROWS} non-missing rows for '{target}'."}

    is_classification = _is_classification_target(working[target])
    if is_classification and working[target].nunique(dropna=True) < 2:
        return {"applicable": False, "reason": f"'{target}' has only one distinct value — nothing to predict."}

    feature_columns = [c for c in working.columns if c != target]
    if not feature_columns:
        return {"applicable": False, "reason": "No other columns available to use as features."}

    X, feature_encoders = _encode_features(working[feature_columns])
    target_encoder = LabelEncoder().fit(working[target].astype(str)) if is_classification else None
    y = target_encoder.transform(working[target].astype(str)) if is_classification else working[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if is_classification else None
    )

    if is_classification:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if is_classification:
        score_name, score = "accuracy", float(accuracy_score(y_test, predictions))
    else:
        score_name, score = "r2", float(r2_score(y_test, predictions))

    ranked = sorted(zip(feature_columns, model.feature_importances_), key=lambda pair: -pair[1])[:max_features]

    return {
        "applicable": True,
        "target": target,
        "task": "classification" if is_classification else "regression",
        "score_name": score_name,
        "score": round(score, 4),
        "n_rows_used": len(working),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": [{"column": c, "importance": round(float(imp), 4)} for c, imp in ranked],
        "model_bundle": {
            "model": model,
            "feature_columns": feature_columns,
            "feature_encoders": feature_encoders,
            "target_encoder": target_encoder,
            "target": target,
            "task": "classification" if is_classification else "regression",
        },
    }


def plot_feature_importance(features: list[dict]) -> str:
    """Horizontal bar chart of feature importances, most important on top. Returns base64 PNG."""
    ordered = list(reversed(features))
    labels = [f["column"] for f in ordered]
    values = [f["importance"] for f in ordered]

    fig, ax = new_figure(figsize=(7, max(2.5, 0.4 * len(ordered))), grid="x")
    ax.barh(labels, values, color=CATEGORICAL[0], edgecolor="none")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=10, colors=INK_SECONDARY)
    fig.tight_layout()
    return fig_to_base64_png(fig)
