"""LangGraph ReAct loop for the regression agent."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from configs import agent_config
from core.llm import get_llm
from core.react_tools import (
    STAT_TOOL_NAMES,
    TOOL_SCHEMAS,
    dumps,
    execute_action,
    prepare_data,
    public_candidate,
    render_final_test,
)
from core.state import ModelState


def _system_prompt(state: ModelState) -> str:
    active = public_candidate(state["active_candidate"]) if state.get("active_candidate") else None
    completed = [public_candidate(item) for item in state.get("completed_candidates", [])]
    discarded = [public_candidate(item) for item in state.get("discarded_attempts", [])]
    return f"""
You are a ReAct multiple-linear-regression agent. Call exactly one tool per turn.
Never calculate or invent statistics yourself; use the statistical tools.

PROJECT BACKGROUND
{state.get('project_background', agent_config.PROJECT_BACKGROUND)}

USER GOAL
{state.get('user_query', '')}

TARGET: {state['target_variable']}
CANDIDATE VARIABLES: {state['candidate_variables']}
REQUIRED COMPLETED CANDIDATES: {agent_config.NUM_COMPLETED_CANDIDATES}

CURRENT ACTIVE CANDIDATE
{dumps(active)}

COMPLETED CANDIDATES
{dumps(completed)}

DISCARDED ATTEMPTS
{dumps(discarded)}

Rules:
- A candidate must run all five tools before acceptance: {sorted(STAT_TOOL_NAMES)}.
- VIF is lower-is-better; values above {agent_config.VIF_WARNING_THRESHOLD} warn of severe collinearity.
- Adjusted R-squared is higher-is-better.
- F-statistic must be interpreted with its p-value.
- Larger absolute t-statistics and coefficient p-values below {agent_config.P_VALUE_THRESHOLD} support significance.
- Validation RMSE is lower-is-better and is comparative; there is no fixed RMSE gate.
- You may abandon a candidate immediately after any unfavorable observation.
- Discarded attempts do not count toward the two completed candidates.
- Feature combinations may not repeat.
- The final test set is unavailable until the best candidate is selected.
- After two candidates are completed, call select_best_candidate with a ranking and concise explanation.
""".strip()


def _tool_call(name: str, args: dict[str, Any], sequence: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"deterministic_call_{sequence}",
                "type": "tool_call",
            }
        ],
    )


def _feature_key(features):
    return tuple(sorted(features))


def _next_features(state: ModelState):
    pool = list(state["candidate_variables"])
    seen = {_feature_key(items) for items in state.get("attempted_feature_sets", [])}
    proposals = []
    if state.get("discarded_attempts"):
        last = state["discarded_attempts"][-1]
        vif = last.get("metrics", {}).get("test_vif", {})
        high = vif.get("high_vif_features", [])
        if high and len(last["selected_variables"]) > 1:
            proposals.append([item for item in last["selected_variables"] if item != high[0]])
    if state.get("completed_candidates"):
        last = state["completed_candidates"][-1]
        t_result = last.get("metrics", {}).get("test_t_stat", {})
        insignificant = t_result.get("insignificant_features", [])
        if insignificant and len(last["selected_variables"]) > 1:
            proposals.append([item for item in last["selected_variables"] if item != insignificant[0]])
        pvalues = t_result.get("feature_pvalues", {})
        if pvalues and len(last["selected_variables"]) > 1:
            weakest = max(pvalues, key=pvalues.get)
            proposals.append([item for item in last["selected_variables"] if item != weakest])
        if len(last["selected_variables"]) > 1:
            proposals.append(last["selected_variables"][:-1])
    proposals.append(pool)
    for size in range(len(pool) - 1, 0, -1):
        proposals.extend([list(group) for group in combinations(pool, size)])
    for proposal in proposals:
        if proposal and _feature_key(proposal) not in seen:
            return proposal
    return None


def _deterministic_decision(state: ModelState) -> AIMessage:
    sequence = state.get("tool_call_count", 0) + 1
    active = state.get("active_candidate")
    completed = state.get("completed_candidates", [])
    if not active:
        if len(completed) >= agent_config.NUM_COMPLETED_CANDIDATES:
            ranked = sorted(
                completed,
                key=lambda item: (
                    item["metrics"]["test_rmse"]["validation_rmse"],
                    -item["metrics"]["test_adjusted_r2"]["adjusted_r2"],
                ),
            )
            best = ranked[0]
            return _tool_call(
                "select_best_candidate",
                {
                    "best_candidate_id": best["candidate_id"],
                    "ranking": [item["candidate_id"] for item in ranked],
                    "comparison_explanation": (
                        "Deterministic ReAct fallback selected the candidate with the lowest "
                        "validation RMSE, using adjusted R² as a tie-breaker."
                    ),
                    "recommended_variables": best["selected_variables"],
                    "risks": ["Selection used the deterministic fallback rather than DeepSeek."],
                },
                sequence,
            )
        features = _next_features(state)
        if not features:
            return _tool_call("inspect_history", {}, sequence)
        return _tool_call(
            "start_candidate",
            {
                "selected_variables": features,
                "decision_summary": "Start the next unique candidate from observed history.",
            },
            sequence,
        )

    candidate_id = active["candidate_id"]
    completed_tools = set(active.get("completed_stat_tools", []))
    if "test_vif" not in completed_tools:
        return _tool_call("test_vif", {"candidate_id": candidate_id}, sequence)
    vif_result = active.get("metrics", {}).get("test_vif", {})
    if (
        vif_result.get("high_vif_features")
        and len(active["selected_variables"]) > 1
    ):
        return _tool_call(
            "abandon_candidate",
            {
                "candidate_id": candidate_id,
                "reason": "VIF screening found severe multicollinearity.",
                "next_feature_plan": f"Remove {vif_result['high_vif_features'][0]} and retry.",
            },
            sequence,
        )
    if active.get("_model") is None:
        return _tool_call("fit_candidate", {"candidate_id": candidate_id}, sequence)
    for tool_name in [
        "test_adjusted_r2",
        "test_f_stat",
        "test_t_stat",
        "test_rmse",
    ]:
        if tool_name not in completed_tools:
            return _tool_call(tool_name, {"candidate_id": candidate_id}, sequence)
    return _tool_call(
        "accept_candidate",
        {
            "candidate_id": candidate_id,
            "decision_summary": "All five required statistical tools completed.",
        },
        sequence,
    )


def build_react_graph(decision_model=None):
    model = decision_model or get_llm()

    def prepare_node(state: ModelState):
        updates = prepare_data(state)
        return {
            **updates,
            "messages": [
                HumanMessage(
                    content=(
                        "Build two distinct OLS candidates using ReAct statistical tools, "
                        "then compare them and select the best model."
                    )
                )
            ],
        }

    def decision_node(state: ModelState):
        if not state.get("use_llm", agent_config.USE_LLM):
            return {"messages": [_deterministic_decision(state)]}
        bound = model.bind_tools(TOOL_SCHEMAS)
        response = bound.invoke([SystemMessage(content=_system_prompt(state)), *state["messages"]])
        return {"messages": [response]}

    def tool_node(state: ModelState):
        message = state["messages"][-1]
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return {
                "messages": [
                    HumanMessage(content="You must call exactly one available tool now.")
                ],
                "tool_call_count": state.get("tool_call_count", 0) + 1,
            }
        if len(tool_calls) > 1:
            return {
                "messages": [
                    ToolMessage(
                        content=dumps({"error": "Call exactly one tool per ReAct turn."}),
                        tool_call_id=call["id"],
                    )
                    for call in tool_calls
                ],
                "tool_call_count": state.get("tool_call_count", 0) + 1,
            }
        call = tool_calls[0]
        updates, observation = execute_action(
            state, call["name"], call.get("args", {}), call["id"]
        )
        return {
            **updates,
            "messages": [ToolMessage(content=dumps(observation), tool_call_id=call["id"])],
        }

    def route_after_tool(state: ModelState):
        if state.get("best_candidate_id"):
            return "finalize"
        if state.get("tool_call_count", 0) >= state.get(
            "max_tool_calls", agent_config.MAX_REACT_TOOL_CALLS
        ):
            return "abort"
        return "continue"

    def final_node(state: ModelState):
        return render_final_test(state)

    def abort_node(state: ModelState):
        return {
            "error": (
                f"ReAct stopped after {state.get('tool_call_count', 0)} tool turns "
                "without selecting a best candidate."
            )
        }

    graph = StateGraph(ModelState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("react_decide", decision_node)
    graph.add_node("execute_tool", tool_node)
    graph.add_node("final_test_and_plot", final_node)
    graph.add_node("abort", abort_node)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "react_decide")
    graph.add_edge("react_decide", "execute_tool")
    graph.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "continue": "react_decide",
            "finalize": "final_test_and_plot",
            "abort": "abort",
        },
    )
    graph.add_edge("final_test_and_plot", END)
    graph.add_edge("abort", END)
    return graph.compile()
