import json
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage

from configs import agent_config
from core.react_tools import STAT_TOOL_NAMES, execute_action, prepare_data
from main import run_agent, save_summary


def _write_linear_csv(path: Path, rows=500):
    rng = np.random.default_rng(20260902)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    category = rng.choice(["A", "B"], size=rows)
    target = 2.0 + 3.5 * x1 - 1.75 * x2 + (category == "B") * 1.2
    target += rng.normal(scale=0.3, size=rows)
    pd.DataFrame(
        {"x1": x1, "x2": x2, "category": category, "target": target}
    ).to_csv(path, index=False)


def test_deterministic_react_builds_two_candidates_and_plot(tmp_path, monkeypatch):
    csv_path = tmp_path / "linear.csv"
    _write_linear_csv(csv_path)
    monkeypatch.setattr(agent_config, "OUTPUT_DIR", str(tmp_path / "outputs"))

    result = run_agent(
        str(csv_path),
        target_variable="target",
        candidate_variables=["x1", "x2", "category"],
        use_llm=False,
    )

    assert not result.get("error")
    assert len(result["completed_candidates"]) == 2
    assert len({tuple(sorted(item["selected_variables"])) for item in result["completed_candidates"]}) == 2
    for candidate in result["completed_candidates"]:
        assert set(candidate["completed_stat_tools"]) == STAT_TOOL_NAMES
        assert set(candidate["metrics"]) == STAT_TOOL_NAMES
    assert result["best_candidate_id"] in result["ranking"]
    assert len(result["ranking"]) == 2
    assert Path(result["test_plot_path"]).is_file()
    assert Path(result["test_plot_path"]).suffix == ".svg"
    assert {entry["chosen_tool"] for entry in result["decision_log"]} >= STAT_TOOL_NAMES
    assert len(result["decision_log"]) == len(result["tool_call_history"])

    saved = save_summary(result, tmp_path / "saved_result")
    saved_payload = json.loads(saved.read_text(encoding="utf-8"))
    assert saved.name == "agent_result.json"
    assert saved_payload["best_candidate_id"] == result["best_candidate_id"]


def test_high_vif_can_immediately_discard_candidate(tmp_path, monkeypatch):
    rng = np.random.default_rng(77)
    rows = 600
    x1 = rng.normal(size=rows)
    x_copy = 2.0 * x1
    noise = rng.normal(size=rows)
    target = 4.0 * x1 + rng.normal(scale=0.2, size=rows)
    csv_path = tmp_path / "collinear.csv"
    pd.DataFrame(
        {"x1": x1, "x_copy": x_copy, "noise": noise, "target": target}
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(agent_config, "OUTPUT_DIR", str(tmp_path / "outputs"))

    result = run_agent(
        str(csv_path),
        target_variable="target",
        candidate_variables=["x1", "x_copy", "noise"],
        use_llm=False,
    )

    assert result["discarded_attempts"]
    rejected = result["discarded_attempts"][0]
    assert rejected["completed_stat_tools"] == ["test_vif"]
    assert rejected["metrics"]["test_vif"]["high_vif_features"]
    assert [item["tool_name"] for item in rejected["tool_trace"]] == [
        "start_candidate",
        "test_vif",
        "abandon_candidate",
    ]
    assert len(result["completed_candidates"]) == 2


def test_accept_rejected_until_all_stat_tools_run(tmp_path):
    csv_path = tmp_path / "guard.csv"
    _write_linear_csv(csv_path, rows=100)
    state = prepare_data(
        {
            "file_location": str(csv_path),
            "target_variable": "target",
            "candidate_variables": ["x1", "x2"],
        }
    )
    updates, _ = execute_action(
        state,
        "start_candidate",
        {"selected_variables": ["x1", "x2"], "decision_summary": "test"},
        "call_start",
    )
    state.update(updates)
    updates, observation = execute_action(
        state,
        "accept_candidate",
        {"candidate_id": 1, "decision_summary": "too early"},
        "call_accept",
    )

    assert observation["error"] == "ValueError"
    assert "missing tools" in observation["message"]
    assert not updates["completed_candidates"]


class ScriptedToolModel:
    def __init__(self, actions):
        self.actions = list(actions)
        self.schemas = None

    def bind_tools(self, schemas):
        self.schemas = schemas
        return self

    def invoke(self, _messages):
        action = self.actions.pop(0)
        sequence = len(self.actions)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": action[0],
                    "args": action[1],
                    "id": f"scripted_{sequence}",
                    "type": "tool_call",
                }
            ],
        )


def test_llm_path_executes_real_tool_calls_and_comparison(tmp_path, monkeypatch):
    csv_path = tmp_path / "scripted.csv"
    _write_linear_csv(csv_path)
    monkeypatch.setattr(agent_config, "OUTPUT_DIR", str(tmp_path / "outputs"))
    actions = []
    for candidate_id, features in [
        (1, ["x1", "x2", "category"]),
        (2, ["x1", "x2"]),
    ]:
        actions.extend(
            [
                ("start_candidate", {"selected_variables": features, "decision_summary": "scripted"}),
                ("test_vif", {"candidate_id": candidate_id}),
                ("fit_candidate", {"candidate_id": candidate_id}),
                ("test_adjusted_r2", {"candidate_id": candidate_id}),
                ("test_f_stat", {"candidate_id": candidate_id}),
                ("test_t_stat", {"candidate_id": candidate_id}),
                ("test_rmse", {"candidate_id": candidate_id}),
                ("accept_candidate", {"candidate_id": candidate_id, "decision_summary": "complete"}),
            ]
        )
    actions.append(
        (
            "select_best_candidate",
            {
                "best_candidate_id": 1,
                "ranking": [1, 2],
                "comparison_explanation": "Candidate 1 has better validation performance.",
                "recommended_variables": ["x1", "x2", "category"],
                "risks": ["OLS assumes linear relationships."],
            },
        )
    )
    model = ScriptedToolModel(actions)

    result = run_agent(
        str(csv_path),
        target_variable="target",
        candidate_variables=["x1", "x2", "category"],
        use_llm=True,
        decision_model=model,
    )

    assert model.schemas
    assert result["best_candidate_id"] == 1
    assert result["ranking"] == [1, 2]
    assert result["comparison_mode"] == "llm"
    assert result["comparison_explanation"].startswith("Candidate 1")
    assert len(result["tool_call_history"]) == 17
