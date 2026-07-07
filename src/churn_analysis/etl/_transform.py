"""Transform data for churn analysis pipeline."""

from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pandas as pd


def replace_missing_values_in_chunks(
    *,
    raw_data_path: Path,
    extraction_func: Callable[[Path], Generator[pd.DataFrame, None, None]],
    col_title: str,
) -> Generator[pd.DataFrame, None, None]:
    """Replace missing values in a specific column with the overall dataset median.

    Args:
        raw_data_path: Absolute or relative path to the raw CSV file.
        extraction_func: Funtion that exctract, validate and yield dataframe chunks.
        col_title: Name of the column where missing values will be replaced.

    Yields:
        Dataframe chunks with missing values replaced by the computed median.
    """
    all_total_charges = []
    for chunk_df in extraction_func(raw_data_path):
        all_total_charges.extend(chunk_df[col_title].dropna(inplace=False).tolist())
    total_charges_median = np.median(all_total_charges)

    for chunk_df in extraction_func(raw_data_path):
        chunk_df[col_title] = chunk_df[col_title].fillna(total_charges_median)
        yield chunk_df
