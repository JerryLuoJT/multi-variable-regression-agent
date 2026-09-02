"""Shared state passed between the regression agent's LangGraph nodes."""

from typing import Any, Dict, List, TypedDict


class ModelState(TypedDict, total=False):
    user_query: str
    file_location: str
    data_location: str  # Backwards-compatible alias.
    target_variable: str
    selected_variables: List[str]
    use_llm: bool

    data_frame: Any
    train_data: Any
    validation_data: Any
    test_data: Any
    column_info: Dict[str, str]
    context_analysis: str

    fitted_model: Any
    feature_columns: List[str]
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any

    metrics: Dict[str, float]
    coefficient_table: Any
    feedback: str
    mse_threshold: float
    passed_mse_gate: bool
    candidate_passed: bool
    retry_exhausted: bool
    iteration_count: int
    max_iterations: int
    attempt_history: List[Dict[str, Any]]
    failure_reasons: List[str]
