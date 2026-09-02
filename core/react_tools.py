"""Deterministic tools available to the ReAct decision layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from configs import agent_config


STAT_TOOL_NAMES = {
    "test_adjusted_r2",
    "test_f_stat",
    "test_t_stat",
    "test_rmse",
    "test_vif",
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "Inspect the prepared dataset schema and candidate variables.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_candidate",
            "description": "Start a new unique regression candidate using a chosen feature subset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "decision_summary": {"type": "string"},
                },
                "required": ["selected_variables", "decision_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_candidate",
            "description": "Fit OLS for the active candidate. VIF may be tested before fitting.",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "integer"}},
                "required": ["candidate_id"],
            },
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "integer"}},
                    "required": ["candidate_id"],
                },
            },
        }
        for name, description in [
            ("test_vif", "Calculate VIF for every active-candidate feature."),
            ("test_adjusted_r2", "Return the fitted model's adjusted R-squared."),
            ("test_f_stat", "Return the fitted model's F-statistic and p-value."),
            ("test_t_stat", "Return coefficient, standard error, t-statistic and p-value."),
            ("test_rmse", "Return training and validation RMSE."),
        ]
    ],
    {
        "type": "function",
        "function": {
            "name": "abandon_candidate",
            "description": "Discard the active candidate immediately after an unfavorable observation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "next_feature_plan": {"type": "string"},
                },
                "required": ["candidate_id", "reason", "next_feature_plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accept_candidate",
            "description": "Accept an active candidate after all five statistical tools ran.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "decision_summary": {"type": "string"},
                },
                "required": ["candidate_id", "decision_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_history",
            "description": "Inspect completed and discarded candidates plus their tool traces.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_best_candidate",
            "description": "Rank exactly two completed candidates and choose the best one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "best_candidate_id": {"type": "integer"},
                    "ranking": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "comparison_explanation": {"type": "string"},
                    "recommended_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "best_candidate_id",
                    "ranking",
                    "comparison_explanation",
                    "recommended_variables",
                    "risks",
                ],
            },
        },
    },
]


def prepare_data(state: dict[str, Any]) -> dict[str, Any]:
    raw_path = state.get("file_location") or agent_config.DATA_FILE
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).drop_duplicates()
    if len(frame) < agent_config.MIN_USABLE_ROWS or len(frame.columns) < 2:
        raise ValueError("Dataset needs at least two columns and enough usable rows")

    target = state.get("target_variable") or agent_config.TARGET_VARIABLE
    if not target:
        query = state.get("user_query", "").lower()
        mentioned = [column for column in frame.columns if str(column).lower() in query]
        target = mentioned[-1] if mentioned else frame.columns[-1]
    if target not in frame.columns:
        raise ValueError(f"Unknown target '{target}'")

    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target])
    if len(frame) < agent_config.MIN_USABLE_ROWS:
        raise ValueError("Too few rows remain after removing missing target values")

    candidate_variables = state.get("candidate_variables")
    if not candidate_variables:
        candidate_variables = [column for column in frame.columns if column != target]
    invalid = [column for column in candidate_variables if column not in frame.columns]
    if invalid or target in candidate_variables:
        raise ValueError(f"Invalid candidate variables: {invalid}")

    development, test = train_test_split(
        frame, test_size=agent_config.TEST_SIZE, random_state=agent_config.RANDOM_STATE
    )
    train, validation = train_test_split(
        development,
        test_size=agent_config.VALIDATION_SIZE_WITHIN_DEVELOPMENT,
        random_state=agent_config.RANDOM_STATE,
    )
    return {
        "file_location": str(path),
        "target_variable": target,
        "candidate_variables": list(candidate_variables),
        "data_frame": frame,
        "train_data": train.copy(),
        "validation_data": validation.copy(),
        "test_data": test.copy(),
        "column_info": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "active_candidate": None,
        "completed_candidates": [],
        "discarded_attempts": [],
        "attempted_feature_sets": [],
        "total_attempts": 0,
        "tool_call_history": [],
        "decision_log": [],
        "tool_call_count": 0,
        "max_tool_calls": agent_config.MAX_REACT_TOOL_CALLS,
    }


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else "Infinity"
    return value


def dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def _feature_key(features: list[str]) -> list[str]:
    return sorted(dict.fromkeys(features))


def _fill_feature(train: pd.Series, others: list[pd.Series]):
    if pd.api.types.is_numeric_dtype(train):
        if agent_config.NUMERIC_IMPUTATION == "mean":
            value = train.mean()
        elif agent_config.NUMERIC_IMPUTATION == "median":
            value = train.median()
        else:
            raise ValueError("NUMERIC_IMPUTATION must be 'mean' or 'median'")
        value = 0.0 if pd.isna(value) else value
    else:
        if agent_config.CATEGORICAL_IMPUTATION == "constant":
            value = agent_config.MISSING_CATEGORY_TOKEN
        elif agent_config.CATEGORICAL_IMPUTATION == "most_frequent":
            modes = train.mode(dropna=True)
            value = agent_config.MISSING_CATEGORY_TOKEN if modes.empty else modes.iloc[0]
        else:
            raise ValueError(
                "CATEGORICAL_IMPUTATION must be 'constant' or 'most_frequent'"
            )
    return train.fillna(value), [series.fillna(value) for series in others]


def _prepare_matrices(state: dict[str, Any], candidate: dict[str, Any]):
    if candidate.get("_X_train") is not None:
        return
    train_parts, validation_parts, test_parts = [], [], []
    feature_map: dict[str, list[str]] = {}
    for feature in candidate["selected_variables"]:
        train_series = state["train_data"][feature].copy()
        validation_series = state["validation_data"][feature].copy()
        test_series = state["test_data"][feature].copy()
        train_series, [validation_series, test_series] = _fill_feature(
            train_series, [validation_series, test_series]
        )
        if pd.api.types.is_numeric_dtype(train_series):
            train_part = pd.DataFrame({feature: train_series.astype(float)})
            validation_part = pd.DataFrame({feature: validation_series.astype(float)})
            test_part = pd.DataFrame({feature: test_series.astype(float)})
        else:
            train_part = pd.get_dummies(
                train_series,
                prefix=feature,
                drop_first=agent_config.ONE_HOT_DROP_FIRST,
                dtype=float,
            )
            validation_part = pd.get_dummies(
                validation_series,
                prefix=feature,
                drop_first=agent_config.ONE_HOT_DROP_FIRST,
                dtype=float,
            ).reindex(columns=train_part.columns, fill_value=0.0)
            test_part = pd.get_dummies(
                test_series,
                prefix=feature,
                drop_first=agent_config.ONE_HOT_DROP_FIRST,
                dtype=float,
            ).reindex(columns=train_part.columns, fill_value=0.0)
        feature_map[feature] = list(train_part.columns)
        train_parts.append(train_part)
        validation_parts.append(validation_part)
        test_parts.append(test_part)

    X_train = pd.concat(train_parts, axis=1)
    X_validation = pd.concat(validation_parts, axis=1).reindex(
        columns=X_train.columns, fill_value=0.0
    )
    X_test = pd.concat(test_parts, axis=1).reindex(
        columns=X_train.columns, fill_value=0.0
    )
    if X_train.shape[1] == 0:
        raise ValueError("No encoded predictors remain")
    candidate["_X_train_raw"] = X_train
    candidate["_X_train"] = sm.add_constant(X_train, has_constant="add")
    candidate["_X_validation"] = sm.add_constant(X_validation, has_constant="add")
    candidate["_X_test"] = sm.add_constant(X_test, has_constant="add")
    candidate["_y_train"] = state["train_data"][state["target_variable"]].astype(float)
    candidate["_y_validation"] = state["validation_data"][state["target_variable"]].astype(float)
    candidate["_y_test"] = state["test_data"][state["target_variable"]].astype(float)
    candidate["_feature_map"] = feature_map


def _active(state: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    candidate = state.get("active_candidate")
    if not candidate or candidate["candidate_id"] != int(candidate_id):
        raise ValueError(f"Candidate {candidate_id} is not active")
    return candidate


def _fitted(state: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    candidate = _active(state, candidate_id)
    if candidate.get("_model") is None:
        raise ValueError("fit_candidate must run before this statistical tool")
    return candidate


def _aggregate_by_feature(candidate: dict[str, Any], encoded_values: dict[str, float]):
    aggregated = {}
    for feature, encoded_columns in candidate["_feature_map"].items():
        values = [encoded_values[column] for column in encoded_columns if column in encoded_values]
        aggregated[feature] = max(values) if values else 0.0
    return aggregated


def _execute_action(state: dict[str, Any], name: str, args: dict[str, Any]):
    if name == "inspect_dataset":
        return {
            "rows": len(state["data_frame"]),
            "target_variable": state["target_variable"],
            "candidate_variables": state["candidate_variables"],
            "column_info": state["column_info"],
            "completed_candidates": len(state["completed_candidates"]),
            "required_completed_candidates": agent_config.NUM_COMPLETED_CANDIDATES,
        }

    if name == "inspect_history":
        return {
            "completed_candidates": [public_candidate(item) for item in state["completed_candidates"]],
            "discarded_attempts": [public_candidate(item) for item in state["discarded_attempts"]],
        }

    if name == "start_candidate":
        if state.get("active_candidate"):
            raise ValueError("Abandon or accept the active candidate first")
        if len(state["completed_candidates"]) >= agent_config.NUM_COMPLETED_CANDIDATES:
            raise ValueError("Two completed candidates already exist; compare them now")
        if state["total_attempts"] >= agent_config.MAX_TOTAL_ATTEMPTS:
            raise ValueError("Maximum total candidate attempts reached")
        features = list(dict.fromkeys(args.get("selected_variables") or []))
        invalid = [feature for feature in features if feature not in state["candidate_variables"]]
        if not features or invalid:
            raise ValueError(f"Invalid selected_variables: {invalid}")
        key = _feature_key(features)
        if key in state["attempted_feature_sets"]:
            raise ValueError("This feature combination has already been attempted")
        candidate_id = state["total_attempts"] + 1
        state["active_candidate"] = {
            "candidate_id": candidate_id,
            "status": "active",
            "selected_variables": features,
            "selection_reason": args.get("decision_summary", ""),
            "metrics": {},
            "completed_stat_tools": [],
            "tool_trace": [],
        }
        state["total_attempts"] = candidate_id
        state["attempted_feature_sets"].append(key)
        return {"candidate_id": candidate_id, "status": "active", "selected_variables": features}

    candidate_id = int(args.get("candidate_id", -1))

    if name == "fit_candidate":
        candidate = _active(state, candidate_id)
        _prepare_matrices(state, candidate)
        model = sm.OLS(candidate["_y_train"], candidate["_X_train"]).fit()
        candidate["_model"] = model
        rank = int(np.linalg.matrix_rank(candidate["_X_train"].to_numpy()))
        columns = int(candidate["_X_train"].shape[1])
        candidate["fit_summary"] = {
            "n_observations": int(model.nobs),
            "n_parameters": columns,
            "matrix_rank": rank,
            "singular": rank < columns,
        }
        return {"candidate_id": candidate_id, "fitted": True, **candidate["fit_summary"]}

    if name == "test_vif":
        candidate = _active(state, candidate_id)
        _prepare_matrices(state, candidate)
        design = sm.add_constant(candidate["_X_train_raw"], has_constant="add")
        encoded_vif = {}
        values = design.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for index, column in enumerate(design.columns):
                if column == "const":
                    continue
                try:
                    value = float(variance_inflation_factor(values, index))
                except Exception:
                    value = math.inf
                encoded_vif[column] = value
        feature_vif = _aggregate_by_feature(candidate, encoded_vif)
        high = [
            feature
            for feature, value in feature_vif.items()
            if not math.isfinite(value) or value > agent_config.VIF_WARNING_THRESHOLD
        ]
        result = {
            "feature_vif": feature_vif,
            "max_vif": max(feature_vif.values(), default=0.0),
            "high_vif_features": high,
            "severity": "high" if high else "normal",
            "recommended_action": "consider_abandon_or_remove_features" if high else "continue",
        }
    elif name == "test_adjusted_r2":
        candidate = _fitted(state, candidate_id)
        result = {"adjusted_r2": float(candidate["_model"].rsquared_adj)}
    elif name == "test_f_stat":
        candidate = _fitted(state, candidate_id)
        f_stat = float(candidate["_model"].fvalue)
        f_pvalue = float(candidate["_model"].f_pvalue)
        result = {
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "model_significant": bool(f_pvalue < agent_config.P_VALUE_THRESHOLD),
        }
    elif name == "test_t_stat":
        candidate = _fitted(state, candidate_id)
        model = candidate["_model"]
        rows = []
        encoded_pvalues = {}
        for variable in model.params.index:
            pvalue = float(model.pvalues[variable])
            rows.append(
                {
                    "variable": str(variable),
                    "coefficient": float(model.params[variable]),
                    "std_error": float(model.bse[variable]),
                    "t_stat": float(model.tvalues[variable]),
                    "p_value": pvalue,
                    "significant": bool(pvalue < agent_config.P_VALUE_THRESHOLD),
                }
            )
            if variable != "const":
                encoded_pvalues[str(variable)] = pvalue
        feature_pvalues = _aggregate_by_feature(candidate, encoded_pvalues)
        insignificant = [
            feature
            for feature, pvalue in feature_pvalues.items()
            if pvalue >= agent_config.P_VALUE_THRESHOLD
        ]
        result = {
            "coefficients": rows,
            "feature_pvalues": feature_pvalues,
            "insignificant_features": insignificant,
            "significant_feature_ratio": float(
                (len(feature_pvalues) - len(insignificant)) / max(len(feature_pvalues), 1)
            ),
        }
    elif name == "test_rmse":
        candidate = _fitted(state, candidate_id)
        model = candidate["_model"]
        train_rmse = float(
            np.sqrt(mean_squared_error(candidate["_y_train"], model.predict(candidate["_X_train"])))
        )
        validation_rmse = float(
            np.sqrt(
                mean_squared_error(
                    candidate["_y_validation"], model.predict(candidate["_X_validation"])
                )
            )
        )
        result = {"train_rmse": train_rmse, "validation_rmse": validation_rmse}
    elif name == "abandon_candidate":
        candidate = _active(state, candidate_id)
        candidate["status"] = "discarded"
        candidate["discard_reason"] = args.get("reason", "")
        candidate["next_feature_plan"] = args.get("next_feature_plan", "")
        state["discarded_attempts"].append(candidate)
        state["active_candidate"] = None
        return {
            "candidate_id": candidate_id,
            "status": "discarded",
            "reason": candidate["discard_reason"],
            "next_feature_plan": candidate["next_feature_plan"],
        }
    elif name == "accept_candidate":
        candidate = _active(state, candidate_id)
        missing = sorted(STAT_TOOL_NAMES - set(candidate["completed_stat_tools"]))
        if missing:
            raise ValueError(f"Candidate statistics incomplete; missing tools: {missing}")
        if candidate.get("_model") is None:
            raise ValueError("Candidate must be fitted before acceptance")
        candidate["status"] = "completed"
        candidate["acceptance_reason"] = args.get("decision_summary", "")
        state["completed_candidates"].append(candidate)
        state["active_candidate"] = None
        return {
            "candidate_id": candidate_id,
            "status": "completed",
            "completed_candidates": len(state["completed_candidates"]),
            "required_completed_candidates": agent_config.NUM_COMPLETED_CANDIDATES,
        }
    elif name == "select_best_candidate":
        if state.get("active_candidate"):
            raise ValueError("Resolve the active candidate before comparison")
        completed = state["completed_candidates"]
        if len(completed) != agent_config.NUM_COMPLETED_CANDIDATES:
            raise ValueError("Exactly two completed candidates are required")
        ids = [candidate["candidate_id"] for candidate in completed]
        ranking = [int(item) for item in args.get("ranking", [])]
        best_id = int(args.get("best_candidate_id", -1))
        if sorted(ranking) != sorted(ids) or len(ranking) != len(set(ranking)):
            raise ValueError(f"ranking must contain each completed candidate once: {ids}")
        if not ranking or ranking[0] != best_id:
            raise ValueError("ranking[0] must equal best_candidate_id")
        best = next(candidate for candidate in completed if candidate["candidate_id"] == best_id)
        recommended = list(args.get("recommended_variables") or [])
        if not recommended or any(item not in best["selected_variables"] for item in recommended):
            raise ValueError("recommended_variables must be a non-empty subset of the best candidate")
        state["best_candidate_id"] = best_id
        state["ranking"] = ranking
        state["comparison_explanation"] = str(args.get("comparison_explanation", ""))
        state["recommended_variables"] = recommended
        state["risks"] = list(args.get("risks") or [])
        state["comparison_mode"] = "llm" if state.get("use_llm") else "deterministic_react"
        return {"best_candidate_id": best_id, "ranking": ranking, "accepted": True}
    else:
        raise ValueError(f"Unknown tool: {name}")

    candidate["metrics"][name] = result
    if name not in candidate["completed_stat_tools"]:
        candidate["completed_stat_tools"].append(name)
    return {"candidate_id": candidate_id, "metric": name, **result}


def execute_action(state: dict[str, Any], name: str, args: dict[str, Any], call_id: str):
    work = dict(state)
    for key in [
        "completed_candidates",
        "discarded_attempts",
        "attempted_feature_sets",
        "tool_call_history",
        "decision_log",
    ]:
        work[key] = list(state.get(key, []))
    if state.get("active_candidate"):
        work["active_candidate"] = dict(state["active_candidate"])
        work["active_candidate"]["metrics"] = dict(state["active_candidate"].get("metrics", {}))
        work["active_candidate"]["completed_stat_tools"] = list(
            state["active_candidate"].get("completed_stat_tools", [])
        )
        work["active_candidate"]["tool_trace"] = list(
            state["active_candidate"].get("tool_trace", [])
        )

    status = "success"
    try:
        observation = _execute_action(work, name, args)
    except Exception as exc:
        status = "error"
        observation = {"error": type(exc).__name__, "message": str(exc)}

    sequence = int(state.get("tool_call_count", 0)) + 1
    candidate_id = args.get("candidate_id") or observation.get("candidate_id")
    log_entry = {
        "sequence": sequence,
        "call_id": call_id,
        "candidate_id": candidate_id,
        "tool_name": name,
        "arguments": args,
        "status": status,
        "observation": observation,
    }
    work["tool_call_history"].append(log_entry)
    work["decision_log"].append(
        {
            "sequence": sequence,
            "candidate_id": candidate_id,
            "chosen_tool": name,
            "decision_summary": args.get("decision_summary")
            or args.get("reason")
            or "",
            "call_id": call_id,
        }
    )

    candidates = []
    if work.get("active_candidate"):
        candidates.append(work["active_candidate"])
    candidates.extend(work["completed_candidates"])
    candidates.extend(work["discarded_attempts"])
    for candidate in candidates:
        if candidate_id and candidate["candidate_id"] == int(candidate_id):
            trace = list(candidate.get("tool_trace", []))
            trace.append(
                {"sequence": sequence, "call_id": call_id, "tool_name": name, "status": status}
            )
            candidate["tool_trace"] = trace
            break

    return {
        "active_candidate": work.get("active_candidate"),
        "completed_candidates": work["completed_candidates"],
        "discarded_attempts": work["discarded_attempts"],
        "attempted_feature_sets": work["attempted_feature_sets"],
        "total_attempts": work.get("total_attempts", state.get("total_attempts", 0)),
        "tool_call_history": work["tool_call_history"],
        "decision_log": work["decision_log"],
        "tool_call_count": sequence,
        "best_candidate_id": work.get("best_candidate_id"),
        "ranking": work.get("ranking"),
        "comparison_explanation": work.get("comparison_explanation"),
        "recommended_variables": work.get("recommended_variables"),
        "risks": work.get("risks"),
        "comparison_mode": work.get("comparison_mode"),
    }, observation


def render_final_test(state: dict[str, Any]) -> dict[str, Any]:
    best = next(
        candidate
        for candidate in state["completed_candidates"]
        if candidate["candidate_id"] == state["best_candidate_id"]
    )
    model = best["_model"]
    predictions = model.predict(best["_X_test"])
    actual = best["_y_test"]
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    r2 = float(r2_score(actual, predictions))

    output_dir = Path(state.get("result_dir") or agent_config.OUTPUT_DIR).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "best_model_test_diagnostics.svg"

    residuals = np.asarray(actual) - np.asarray(predictions)

    def scale(values, start, length, invert=False):
        values = np.asarray(values, dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        span = high - low or 1.0
        normalized = (values - low) / span
        if invert:
            normalized = 1.0 - normalized
        return start + normalized * length, low, high

    width, height = 1200, 520
    panel_width, panel_height = 500, 350
    left_x, right_x, top = 70, 650, 95
    all_values = np.concatenate([np.asarray(actual), np.asarray(predictions)])
    all_x, value_low, value_high = scale(all_values, left_x, panel_width)
    all_y, _, _ = scale(all_values, top, panel_height, invert=True)
    actual_x = all_x[: len(actual)]
    predicted_y = all_y[len(actual) :]
    prediction_x, _, _ = scale(predictions, right_x, panel_width)
    residual_y, residual_low, residual_high = scale(
        residuals, top, panel_height, invert=True
    )
    residual_span = residual_high - residual_low or 1.0
    zero_y = top + (1.0 - (0.0 - residual_low) / residual_span) * panel_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.label{font-size:14px}.metric{font-size:14px;font-weight:600}</style>',
        f'<text x="600" y="34" text-anchor="middle" class="title">Best candidate #{best["candidate_id"]} test diagnostics</text>',
        f'<text x="{left_x + panel_width/2}" y="70" text-anchor="middle" class="title">Actual vs Predicted</text>',
        f'<text x="{right_x + panel_width/2}" y="70" text-anchor="middle" class="title">Residuals vs Predicted</text>',
        f'<rect x="{left_x}" y="{top}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#475569"/>',
        f'<rect x="{right_x}" y="{top}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#475569"/>',
        f'<line x1="{left_x}" y1="{top + panel_height}" x2="{left_x + panel_width}" y2="{top}" stroke="#ef4444" stroke-dasharray="7 5"/>',
        f'<line x1="{right_x}" y1="{zero_y:.2f}" x2="{right_x + panel_width}" y2="{zero_y:.2f}" stroke="#ef4444" stroke-dasharray="7 5"/>',
    ]
    for x, y in zip(actual_x, predicted_y):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2563eb" fill-opacity="0.65"/>')
    for x, y in zip(prediction_x, residual_y):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#0f766e" fill-opacity="0.65"/>')
    parts.extend(
        [
            f'<text x="{left_x + panel_width/2}" y="478" text-anchor="middle" class="label">Actual</text>',
            f'<text x="{right_x + panel_width/2}" y="478" text-anchor="middle" class="label">Predicted</text>',
            f'<text x="{left_x}" y="505" class="metric">Test RMSE: {rmse:.4f} · Test R²: {r2:.4f} · N: {len(actual)}</text>',
            f'<text x="{right_x}" y="505" class="metric">Residual range: {residual_low:.4f} to {residual_high:.4f}</text>',
            f'<text x="{left_x}" y="92" class="label">{value_high:.3g}</text>',
            f'<text x="{left_x}" y="{top + panel_height - 5}" class="label">{value_low:.3g}</text>',
            f'<text x="{right_x}" y="92" class="label">{residual_high:.3g}</text>',
            f'<text x="{right_x}" y="{top + panel_height - 5}" class="label">{residual_low:.3g}</text>',
            '</svg>',
        ]
    )
    plot_path.write_text("\n".join(parts), encoding="utf-8")

    return {
        "final_test_metrics": {"rmse": rmse, "r2": r2, "n_samples": int(len(actual))},
        "test_plot_path": str(plot_path),
    }
