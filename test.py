"""Quick local smoke test; the assertions live in test/test_regression_agent.py."""

from main import run_agent


if __name__ == "__main__":
    result = run_agent(
        "test/test.csv",
        target_variable="target",
        selected_variables=["x1", "x2", "category"],
    )
    print(result["feedback"])
    if not result["passed_mse_gate"]:
        raise SystemExit(2)
