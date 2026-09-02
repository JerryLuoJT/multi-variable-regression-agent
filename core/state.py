"""State schema for the tool-driven ReAct regression agent."""

from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph.message import add_messages


class ModelState(TypedDict, total=False):
    messages: Annotated[List[Any], add_messages]
    project_background: str
    user_query: str
    file_location: str
    target_variable: str
    candidate_variables: List[str]
    use_llm: bool
    result_dir: str

    data_frame: Any
    train_data: Any
    validation_data: Any
    test_data: Any
    column_info: Dict[str, str]

    active_candidate: Any
    completed_candidates: List[Dict[str, Any]]
    discarded_attempts: List[Dict[str, Any]]
    attempted_feature_sets: List[List[str]]
    total_attempts: int

    tool_call_history: List[Dict[str, Any]]
    decision_log: List[Dict[str, Any]]
    tool_call_count: int
    max_tool_calls: int

    best_candidate_id: int
    ranking: List[int]
    comparison_explanation: str
    recommended_variables: List[str]
    risks: List[str]
    comparison_mode: str

    final_test_metrics: Dict[str, float]
    test_plot_path: str
    error: str
