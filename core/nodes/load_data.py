from pathlib import Path

import pandas as pd

from configs.agent_config import MIN_USABLE_ROWS
from core.state import ModelState


def load_data_node(state: ModelState):
    """Load a non-empty CSV file and expose its schema to later nodes."""
    raw_path = state.get("file_location") or state.get("data_location")
    if not raw_path:
        raise ValueError("file_location is required")

    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV input is supported: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file contains no rows: {path}")
    if len(df.columns) < 2:
        raise ValueError("Regression requires a target and at least one predictor")
    if len(df) < MIN_USABLE_ROWS:
        raise ValueError(
            f"CSV requires at least {MIN_USABLE_ROWS} rows; received {len(df)}"
        )

    return {
        "file_location": str(path),
        "data_frame": df,
        "column_info": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
