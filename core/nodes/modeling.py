import pandas as pd
import statsmodels.api as sm

from configs.agent_config import ONE_HOT_DROP_FIRST
from core.state import ModelState


def model_generation_node(state: ModelState):
    """Fit an ordinary least-squares multiple linear regression."""
    target = state["target_variable"]
    predictors = state["selected_variables"]
    train = state["train_data"]
    validation = state["validation_data"]
    test = state["test_data"]

    X_train = pd.get_dummies(
        train[predictors], drop_first=ONE_HOT_DROP_FIRST, dtype=float
    )
    X_validation = pd.get_dummies(
        validation[predictors], drop_first=ONE_HOT_DROP_FIRST, dtype=float
    )
    X_test = pd.get_dummies(
        test[predictors], drop_first=ONE_HOT_DROP_FIRST, dtype=float
    )
    X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    if X_train.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after encoding")

    X_train = sm.add_constant(X_train, has_constant="add")
    X_validation = sm.add_constant(X_validation, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")
    y_train = train[target].astype(float)
    y_validation = validation[target].astype(float)
    y_test = test[target].astype(float)
    model = sm.OLS(y_train, X_train).fit()

    return {
        "fitted_model": model,
        "feature_columns": list(X_train.columns),
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }
