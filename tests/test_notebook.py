"""Unit tests for reproducible notebook export (no API key required)."""

import json

import pandas as pd
from agent.notebook import build_notebook, notebook_to_json


def _df():
    return pd.DataFrame({"a": range(30), "b": range(30, 60), "target": [i % 2 for i in range(30)]})


def test_build_notebook_has_valid_nbformat_structure():
    notebook = build_notebook(_df(), "sample.csv")
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert len(notebook["cells"]) >= 3


def test_build_notebook_embeds_data_loadable_round_trip():
    df = _df()
    notebook = build_notebook(df, "sample.csv")
    imports_source = "".join(notebook["cells"][1]["source"])
    data_source = "".join(notebook["cells"][2]["source"])
    assert "pd.read_csv" in data_source
    namespace = {}
    exec(imports_source + "\n" + data_source.replace("df.head()", ""), namespace)
    pd.testing.assert_frame_equal(namespace["df"], df)


def test_build_notebook_samples_large_datasets():
    df = pd.DataFrame({"a": range(6000), "b": range(6000)})
    notebook = build_notebook(df, "big.csv")
    data_cell = notebook["cells"][2]
    source = "".join(data_cell["source"])
    assert "Sampled" in source


def test_build_notebook_adds_predict_section_when_applicable():
    predict_result = {"applicable": True, "target": "target", "task": "classification"}
    notebook = build_notebook(_df(), "sample.csv", predict_result=predict_result)
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "RandomForestClassifier" in joined
    assert "'target'" in joined


def test_build_notebook_skips_predict_section_when_not_applicable():
    predict_result = {"applicable": False, "reason": "not enough data"}
    notebook = build_notebook(_df(), "sample.csv", predict_result=predict_result)
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "RandomForestClassifier" not in joined


def test_build_notebook_adds_segment_section_when_applicable():
    segment_result = {"applicable": True, "columns_used": ["a", "b"], "n_clusters": 3}
    notebook = build_notebook(_df(), "sample.csv", segment_result=segment_result)
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "KMeans" in joined


def test_build_notebook_adds_trend_section_when_applicable():
    trend_result = {"applicable": True, "date_column": "d", "metric_column": "a", "agg": "sum"}
    notebook = build_notebook(_df(), "sample.csv", trend_result=trend_result)
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "d" in joined and "Trend" in "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


def test_notebook_to_json_produces_valid_json():
    notebook = build_notebook(_df(), "sample.csv")
    text = notebook_to_json(notebook)
    parsed = json.loads(text)
    assert parsed["nbformat"] == 4
