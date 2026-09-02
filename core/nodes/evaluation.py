import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.agent_config import (
    ENABLE_FEATURE_PRUNING,
    MAX_ITERATIONS,
    MIN_SELECTED_VARIABLES,
    MSE_THRESHOLD,
    P_VALUE_THRESHOLD,
)
from core.state import ModelState


DEFAULT_MSE_THRESHOLD = MSE_THRESHOLD
DEFAULT_MAX_ITERATIONS = MAX_ITERATIONS


def _mse(actual, predicted):
    return float(mean_squared_error(actual, predicted))


def _original_feature_pvalues(state, model):
    grouped = {}
    for feature in state["selected_variables"]:
        values = [
            float(value) if np.isfinite(value) else 1.0
            for name, value in model.pvalues.items()
            if name == feature or str(name).startswith(f"{feature}_")
        ]
        grouped[feature] = max(values) if values else 1.0
    return grouped


def evaluation_node(state: ModelState):
    """Evaluate a candidate on train/validation and record retry memory."""
    model = state["fitted_model"]
    threshold = float(state.get("mse_threshold", DEFAULT_MSE_THRESHOLD))
    train_mse = _mse(state["y_train"], model.predict(state["X_train"]))
    validation_mse = _mse(
        state["y_validation"], model.predict(state["X_validation"])
    )
    feature_pvalues = _original_feature_pvalues(state, model)

    failure_reasons = []
    if train_mse >= threshold:
        failure_reasons.append(
            f"training MSE {train_mse:.6f} is not below {threshold:.6f}"
        )
    if validation_mse >= threshold:
        failure_reasons.append(
            f"validation MSE {validation_mse:.6f} is not below {threshold:.6f}"
        )
    weak_features = sorted(
        [
            feature
            for feature, pvalue in feature_pvalues.items()
            if pvalue > P_VALUE_THRESHOLD
        ],
        key=lambda feature: feature_pvalues[feature],
        reverse=True,
    )
    if (
        ENABLE_FEATURE_PRUNING
        and weak_features
        and len(state["selected_variables"]) > MIN_SELECTED_VARIABLES
    ):
        weakest = weak_features[0]
        failure_reasons.append(
            f"feature '{weakest}' has p-value {feature_pvalues[weakest]:.6f}, "
            f"above {P_VALUE_THRESHOLD:.6f}; retry with a smaller feature set"
        )

    iteration = len(state.get("attempt_history", [])) + 1
    attempt = {
        "iteration": iteration,
        "selected_variables": list(state["selected_variables"]),
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "feature_pvalues": feature_pvalues,
        "weak_features": weak_features,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
    }
    history = list(state.get("attempt_history", [])) + [attempt]
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    candidate_passed = not failure_reasons

    return {
        "attempt_history": history,
        "iteration_count": iteration,
        "failure_reasons": failure_reasons,
        "candidate_passed": candidate_passed,
        "retry_exhausted": not candidate_passed and iteration >= max_iterations,
        "feedback": (
            f"Candidate passed on iteration {iteration}."
            if candidate_passed
            else f"Candidate failed on iteration {iteration}: "
            + "; ".join(failure_reasons)
        ),
    }


def route_after_evaluation(state: ModelState):
    if state["candidate_passed"] or state["retry_exhausted"]:
        return "finalize"
    return "retry"


def final_evaluation_node(state: ModelState):
    """Use the untouched test set once, after tuning has stopped."""
    model = state["fitted_model"]
    predictions = model.predict(state["X_test"])
    test_mse = _mse(state["y_test"], predictions)
    threshold = float(state.get("mse_threshold", DEFAULT_MSE_THRESHOLD))
    passed = bool(state["candidate_passed"] and test_mse < threshold)

    last_attempt = state["attempt_history"][-1]
    metrics = {
        "mse": test_mse,
        "test_mse": test_mse,
        "train_mse": float(last_attempt["train_mse"]),
        "validation_mse": float(last_attempt["validation_mse"]),
        "rmse": float(np.sqrt(test_mse)),
        "mae": float(mean_absolute_error(state["y_test"], predictions)),
        "r2": float(r2_score(state["y_test"], predictions)),
        "adjusted_r2_train": float(model.rsquared_adj),
        "f_test_pvalue": float(model.f_pvalue),
    }
    coefficient_table = pd.DataFrame(
        {
            "variable": model.params.index,
            "coefficient": model.params.values,
            "p_value": model.pvalues.values,
        }
    )
    if passed:
        feedback = (
            f"PASS after {state['iteration_count']} attempt(s): final test MSE "
            f"{test_mse:.6f} is below {threshold:.6f}."
        )
    elif state["retry_exhausted"]:
        feedback = (
            f"FAIL after exhausting {state['iteration_count']} attempt(s). "
            f"Final test MSE is {test_mse:.6f}."
        )
    else:
        feedback = (
            f"FAIL: candidate passed validation but final test MSE {test_mse:.6f} "
            f"is not below {threshold:.6f}."
        )
    return {
        "metrics": metrics,
        "coefficient_table": coefficient_table,
        "mse_threshold": threshold,
        "passed_mse_gate": passed,
        "feedback": feedback,
    }
