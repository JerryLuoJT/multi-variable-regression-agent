"""Quick offline smoke test for the ReAct workflow."""

from main import run_agent, summary


if __name__ == "__main__":
    result = run_agent(
        "test/test.csv",
        target_variable="target",
        candidate_variables=["x1", "x2", "category"],
        use_llm=False,
    )
    print(summary(result))
    if result.get("error"):
        raise SystemExit(1)
