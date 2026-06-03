from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pandas as pd


def replace_missing_values_in_chunks(
    *,
    raw_data_path: Path,
    extraction_func: Callable[[Path], Generator[pd.DataFrame, None, None]],
) -> Generator[pd.DataFrame, None, None]:
    """
    A function that transforms missing values into the overall median of the data.
    """
    all_total_charges = []
    for chunk_df in extraction_func(raw_data_path):
        all_total_charges.extend(
            chunk_df["total_charges"].dropna(inplace=False).tolist()
        )
    total_charges_median = np.median(all_total_charges)

    for chunk_df in extraction_func(raw_data_path):
        chunk_df["total_charges"] = chunk_df["total_charges"].fillna(
            total_charges_median
        )
        yield chunk_df
