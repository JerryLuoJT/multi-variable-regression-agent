import json
import re
from itertools import combinations

from configs.prompts import TASK_UNDERSTANDING_PROMPT
from core.llm import get_llm
from core.state import ModelState


def _parse_json_response(content):
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = re.sub(r"^```(?:json)?|```$", "", str(content).strip()).strip()
    return json.loads(text)


def _feature_key(features):
    return tuple(sorted(features))


def _next_deterministic_features(all_predictors, history):
    """Return the next unseen feature subset using prior failure diagnostics."""
    seen = {_feature_key(attempt["selected_variables"]) for attempt in history}
    last_pvalues = history[-1].get("feature_pvalues", {}) if history else {}
    previous = list(history[-1]["selected_variables"]) if history else []
    previous_removal_order = sorted(
        previous,
        key=lambda feature: last_pvalues.get(feature, -1.0),
        reverse=True,
    )
    # Prefer true backward elimination from the current feature set.
    candidates = [
        [feature for feature in previous if feature != removed]
        for removed in previous_removal_order
    ]

    # If an initial set was too narrow, expansion can recover omitted signal.
    candidates.append(list(all_predictors))
    all_removal_order = sorted(
        all_predictors,
        key=lambda feature: last_pvalues.get(feature, -1.0),
        reverse=True,
    )
    candidates.extend(
        [[feature for feature in all_predictors if feature != removed]
         for removed in all_removal_order]
    )
    candidates.extend([[feature] for feature in all_predictors])

    # Cover additional subsets for wider datasets while max_iterations remains
    # the authoritative bound on actual model fits.
    for size in range(len(all_predictors) - 2, 1, -1):
        candidates.extend([list(group) for group in combinations(all_predictors, size)])

    for candidate in candidates:
        if candidate and _feature_key(candidate) not in seen:
            return candidate
    return None


def variable_selection_node(state: ModelState):
    """Choose the target and predictors, with an optional Gemini assist."""
    df = state["data_frame"]
    columns = list(df.columns)
    target = state.get("target_variable")
    selected = state.get("selected_variables")
    history = list(state.get("attempt_history", []))
    context = "Variables supplied explicitly by the caller."

    if target and target not in columns:
        raise ValueError(f"Unknown target_variable '{target}'. Available: {columns}")

    if history:
        all_predictors = [column for column in columns if column != target]
        selected = None
        if state.get("use_llm"):
            prompt = TASK_UNDERSTANDING_PROMPT.format(
                user_query=state.get("user_query", ""),
                column_info=state["column_info"],
                attempt_history=json.dumps(history, ensure_ascii=False, default=str),
            )
            try:
                result = _parse_json_response(get_llm().invoke(prompt).content)
                llm_features = result.get("selected_variables") or []
                if (
                    llm_features
                    and all(feature in all_predictors for feature in llm_features)
                    and _feature_key(llm_features)
                    not in {_feature_key(item["selected_variables"]) for item in history}
                ):
                    selected = llm_features
                    context = result.get("context_analysis", "Retry selected by Gemini.")
            except Exception:
                # A deterministic retry keeps the graph functional if the LLM is
                # unavailable or returns invalid JSON.
                selected = None
        if not selected:
            selected = _next_deterministic_features(all_predictors, history)
            context = "Variables revised from the recorded failure history."
        if not selected:
            # The router normally prevents this by enforcing max_iterations.
            selected = history[-1]["selected_variables"]

    if not history and state.get("use_llm") and (not target or not selected):
        prompt = TASK_UNDERSTANDING_PROMPT.format(
            user_query=state.get("user_query", ""),
            column_info=state["column_info"],
            attempt_history="[]",
        )
        result = _parse_json_response(get_llm().invoke(prompt).content)
        target = target or result.get("target_variable")
        selected = selected or result.get("selected_variables")
        context = result.get("context_analysis", "Variables selected by Gemini.")

    if not target:
        query = state.get("user_query", "").lower()
        mentioned = [column for column in columns if str(column).lower() in query]
        target = mentioned[-1] if mentioned else columns[-1]
        context = (
            f"Target inferred as '{target}'. Pass target_variable explicitly "
            "when this inference is not intended."
        )

    if target not in columns:
        raise ValueError(f"Unknown target_variable '{target}'. Available: {columns}")

    selected = list(selected) if selected else [column for column in columns if column != target]
    selected = list(dict.fromkeys(selected))
    invalid = [column for column in selected if column not in columns]
    if invalid:
        raise ValueError(f"Unknown selected_variables: {invalid}")
    if target in selected:
        raise ValueError("target_variable cannot also be a predictor")
    if not selected:
        raise ValueError("At least one predictor is required")

    return {
        "target_variable": target,
        "selected_variables": selected,
        "context_analysis": context,
    }
