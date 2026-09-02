import argparse
import json

from langgraph.graph import END, StateGraph

from configs import agent_config
from core.nodes.data_cleaning import data_cleaning_node
from core.nodes.evaluation import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MSE_THRESHOLD,
    evaluation_node,
    final_evaluation_node,
    route_after_evaluation,
)
from core.nodes.load_data import load_data_node
from core.nodes.modeling import model_generation_node
from core.nodes.variable_selection import variable_selection_node
from core.state import ModelState


def build_agent():
    workflow = StateGraph(ModelState)
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("select_variables", variable_selection_node)
    workflow.add_node("clean_and_split", data_cleaning_node)
    workflow.add_node("fit_regression", model_generation_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("final_evaluate", final_evaluation_node)
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "select_variables")
    workflow.add_edge("select_variables", "clean_and_split")
    workflow.add_edge("clean_and_split", "fit_regression")
    workflow.add_edge("fit_regression", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"retry": "select_variables", "finalize": "final_evaluate"},
    )
    workflow.add_edge("final_evaluate", END)
    return workflow.compile()


app = build_agent()


def run_agent(
    file_location,
    user_query=agent_config.USER_QUERY,
    target_variable=None,
    selected_variables=None,
    mse_threshold=DEFAULT_MSE_THRESHOLD,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    use_llm=agent_config.USE_LLM,
):
    inputs = {
        "file_location": file_location,
        "user_query": user_query,
        "mse_threshold": mse_threshold,
        "max_iterations": max_iterations,
        "use_llm": use_llm,
        "attempt_history": [],
    }
    if target_variable:
        inputs["target_variable"] = target_variable
    if selected_variables:
        inputs["selected_variables"] = selected_variables
    return app.invoke(inputs)


def _summary(result):
    return {
        "target_variable": result["target_variable"],
        "selected_variables": result["selected_variables"],
        "metrics": result["metrics"],
        "mse_threshold": result["mse_threshold"],
        "passed_mse_gate": result["passed_mse_gate"],
        "iteration_count": result["iteration_count"],
        "attempt_history": result["attempt_history"],
        "feedback": result["feedback"],
    }


def main():
    parser = argparse.ArgumentParser(description="Multiple linear regression agent")
    parser.add_argument("csv_file", nargs="?", default=agent_config.DATA_FILE)
    parser.add_argument("--query", default=agent_config.USER_QUERY)
    parser.add_argument("--target", default=agent_config.TARGET_VARIABLE)
    parser.add_argument(
        "--features", nargs="+", default=agent_config.SELECTED_VARIABLES
    )
    parser.add_argument("--mse-threshold", type=float, default=DEFAULT_MSE_THRESHOLD)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=agent_config.USE_LLM,
    )
    args = parser.parse_args()

    result = run_agent(
        file_location=args.csv_file,
        user_query=args.query,
        target_variable=args.target,
        selected_variables=args.features,
        mse_threshold=args.mse_threshold,
        max_iterations=args.max_iterations,
        use_llm=args.use_llm,
    )
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed_mse_gate"] else 2)


if __name__ == "__main__":
    main()
