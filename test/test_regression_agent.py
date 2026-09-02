from pathlib import Path

import numpy as np
import pandas as pd

from main import run_agent


def test_held_out_mse_is_below_five(tmp_path: Path):
    rng = np.random.default_rng(20260831)
    rows = 500
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    category = rng.choice(["A", "B"], size=rows)
    noise = rng.normal(scale=0.35, size=rows)
    target = 2.0 + 3.5 * x1 - 1.75 * x2 + (category == "B") * 1.2 + noise
    frame = pd.DataFrame({"x1": x1, "x2": x2, "category": category, "target": target})
    csv_path = tmp_path / "regression.csv"
    frame.to_csv(csv_path, index=False)

    result = run_agent(
        file_location=str(csv_path),
        target_variable="target",
        selected_variables=["x1", "x2", "category"],
        mse_threshold=5.0,
        use_llm=False,
    )

    assert result["passed_mse_gate"] is True
    assert result["metrics"]["mse"] < 5.0
    assert result["iteration_count"] == 1


def test_bad_model_fails_the_gate(tmp_path: Path):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"x": rng.normal(size=200), "target": rng.normal(scale=10, size=200)}
    )
    csv_path = tmp_path / "unpredictable.csv"
    frame.to_csv(csv_path, index=False)

    result = run_agent(
        str(csv_path),
        target_variable="target",
        mse_threshold=5.0,
        max_iterations=3,
        use_llm=False,
    )

    assert result["passed_mse_gate"] is False
    assert result["metrics"]["mse"] >= 5.0
    assert result["iteration_count"] == 3
    assert len(result["attempt_history"]) == 3
    assert all(attempt["failure_reasons"] for attempt in result["attempt_history"])


def test_failed_attempt_returns_to_variable_selection(tmp_path: Path):
    rng = np.random.default_rng(99)
    rows = 600
    weak = rng.normal(size=rows)
    useful = rng.normal(size=rows)
    target = 4.0 * useful + rng.normal(scale=0.25, size=rows)
    frame = pd.DataFrame({"weak": weak, "useful": useful, "target": target})
    csv_path = tmp_path / "retry.csv"
    frame.to_csv(csv_path, index=False)

    result = run_agent(
        str(csv_path),
        target_variable="target",
        selected_variables=["weak"],
        mse_threshold=5.0,
        max_iterations=4,
        use_llm=False,
    )

    assert result["passed_mse_gate"] is True
    assert result["iteration_count"] == 3
    assert result["attempt_history"][0]["passed"] is False
    assert result["attempt_history"][0]["selected_variables"] == ["weak"]
    assert result["attempt_history"][1]["passed"] is False
    assert "useful" in result["attempt_history"][1]["selected_variables"]
    assert "weak" in result["attempt_history"][1]["weak_features"]
    assert result["attempt_history"][2]["passed"] is True
    assert result["selected_variables"] == ["useful"]


def test_insignificant_feature_is_pruned_even_when_mse_passes(tmp_path: Path):
    rng = np.random.default_rng(1234)
    rows = 1000
    signal = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    target = 3.0 * signal + rng.normal(scale=0.2, size=rows)
    frame = pd.DataFrame({"signal": signal, "noise": noise, "target": target})
    csv_path = tmp_path / "pruning.csv"
    frame.to_csv(csv_path, index=False)

    result = run_agent(
        str(csv_path),
        target_variable="target",
        selected_variables=["signal", "noise"],
        mse_threshold=5.0,
        max_iterations=4,
        use_llm=False,
    )

    assert result["passed_mse_gate"] is True
    assert result["iteration_count"] == 2
    assert result["attempt_history"][0]["train_mse"] < 5.0
    assert result["attempt_history"][0]["validation_mse"] < 5.0
    assert "noise" in result["attempt_history"][0]["weak_features"]
    assert result["selected_variables"] == ["signal"]
