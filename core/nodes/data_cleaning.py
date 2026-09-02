import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.agent_config import (
    CATEGORICAL_IMPUTATION,
    MIN_USABLE_ROWS,
    MISSING_CATEGORY_TOKEN,
    NUMERIC_IMPUTATION,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE_WITHIN_DEVELOPMENT,
)
from core.state import ModelState


def _fill_from_training(train, other_sets, columns):
    for column in columns:
        if pd.api.types.is_numeric_dtype(train[column]):
            if NUMERIC_IMPUTATION == "median":
                value = train[column].median()
            elif NUMERIC_IMPUTATION == "mean":
                value = train[column].mean()
            else:
                raise ValueError(
                    "NUMERIC_IMPUTATION must be 'median' or 'mean'"
                )
            value = 0.0 if pd.isna(value) else value
        else:
            if CATEGORICAL_IMPUTATION == "most_frequent":
                modes = train[column].mode(dropna=True)
                value = MISSING_CATEGORY_TOKEN if modes.empty else modes.iloc[0]
            elif CATEGORICAL_IMPUTATION == "constant":
                value = MISSING_CATEGORY_TOKEN
            else:
                raise ValueError(
                    "CATEGORICAL_IMPUTATION must be 'most_frequent' or 'constant'"
                )
        train[column] = train[column].fillna(value)
        for frame in other_sets:
            frame[column] = frame[column].fillna(value)


def data_cleaning_node(state: ModelState):
    """Clean, split, and impute without using test-set statistics."""
    target = state["target_variable"]
    predictors = state["selected_variables"]
    needed = predictors + [target]

    df = state["data_frame"][needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    if len(df) < MIN_USABLE_ROWS:
        raise ValueError(
            f"At least {MIN_USABLE_ROWS} usable rows are required for splitting"
        )

    development, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, validation = train_test_split(
        development,
        test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
        random_state=RANDOM_STATE,
    )
    train, validation, test = train.copy(), validation.copy(), test.copy()
    _fill_from_training(train, [validation, test], predictors)

    return {
        "train_data": train,
        "validation_data": validation,
        "test_data": test,
    }
