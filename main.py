import argparse
import json
from pathlib import Path

from configs import agent_config
from core.react_agent import build_react_graph
from core.react_tools import public_candidate


def build_agent(decision_model=None):
    return build_react_graph(decision_model=decision_model)


def run_agent(
    file_location=agent_config.DATA_FILE,
    user_query=agent_config.USER_QUERY,
    project_background=agent_config.PROJECT_BACKGROUND,
    target_variable=agent_config.TARGET_VARIABLE,
    candidate_variables=agent_config.SELECTED_VARIABLES,
    use_llm=agent_config.USE_LLM,
    result_dir=None,
    decision_model=None,
):
    graph = build_agent(decision_model=decision_model)
    inputs = {
        "file_location": file_location,
        "user_query": user_query,
        "project_background": project_background,
        "target_variable": target_variable,
        "candidate_variables": candidate_variables,
        "use_llm": use_llm,
        "result_dir": result_dir,
    }
    return graph.invoke(
        inputs,
        {"recursion_limit": agent_config.MAX_REACT_TOOL_CALLS * 3 + 10},
    )


def summary(result):
    return {
        "target_variable": result.get("target_variable"),
        "candidate_variables": result.get("candidate_variables"),
        "completed_candidates": [
            public_candidate(candidate)
            for candidate in result.get("completed_candidates", [])
        ],
        "discarded_attempts": [
            public_candidate(candidate)
            for candidate in result.get("discarded_attempts", [])
        ],
        "best_candidate_id": result.get("best_candidate_id"),
        "ranking": result.get("ranking"),
        "comparison_explanation": result.get("comparison_explanation"),
        "recommended_variables": result.get("recommended_variables"),
        "risks": result.get("risks"),
        "comparison_mode": result.get("comparison_mode"),
        "final_test_metrics": result.get("final_test_metrics"),
        "test_plot_path": result.get("test_plot_path"),
        "tool_call_history": result.get("tool_call_history", []),
        "decision_log": result.get("decision_log", []),
        "error": result.get("error"),
    }


def save_summary(result, result_dir, filename="agent_result.json"):
    output_dir = Path(result_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(
        json.dumps(summary(result), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ReAct multiple regression agent")
    parser.add_argument("csv_file", nargs="?", default=agent_config.DATA_FILE)
    parser.add_argument("--query", default=agent_config.USER_QUERY)
    parser.add_argument("--background", default=agent_config.PROJECT_BACKGROUND)
    parser.add_argument("--target", default=agent_config.TARGET_VARIABLE)
    parser.add_argument(
        "--features", nargs="+", default=agent_config.SELECTED_VARIABLES
    )
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=agent_config.USE_LLM,
    )
    parser.add_argument(
        "--result-dir",
        help="Write the final plot and agent_result.json into this directory.",
    )
    args = parser.parse_args()
    result = run_agent(
        file_location=args.csv_file,
        user_query=args.query,
        project_background=args.background,
        target_variable=args.target,
        candidate_variables=args.features,
        use_llm=args.use_llm,
        result_dir=args.result_dir,
    )
    rendered = json.dumps(summary(result), ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.result_dir:
        save_summary(result, args.result_dir)
    raise SystemExit(1 if result.get("error") else 0)


if __name__ == "__main__":
    main()
