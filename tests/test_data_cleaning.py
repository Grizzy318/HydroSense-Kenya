import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import remove_duplicates, fill_missing_values, validate_ranges


# ---------------------------------------------------------------------------
# Data Cleaning & Validation Tests
# ---------------------------------------------------------------------------

def test_remove_duplicates():
    """Ensures duplicate rows are removed from the dataset."""
    df = pd.DataFrame({
        'date': ['2026-03-01', '2026-03-01', '2026-03-02'], 
        'val': [1, 1, 2]
    })
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2


def test_fill_missing_values():
    """Validates that missing (NaN) numeric values are imputed using the column mean."""
    df = pd.DataFrame({
        'temp': [20.0, np.nan, 30.0], 
        'status': ['OK', 'OK', 'OK']
    })
    cleaned = fill_missing_values(df)
    assert cleaned['temp'].isna().sum() == 0
    assert cleaned['temp'].iloc[1] == 25.0  


def test_validate_ranges():
    """Ensures extreme outliers (e.g., sensor glitches) outside physical bounds are dropped."""
    df = pd.DataFrame({'temperature_c': [15, 25, -20, 35, 80]})
    cleaned = validate_ranges(df, 'temperature_c', minimum=-10, maximum=60)
    
    assert len(cleaned) == 3
    assert cleaned['temperature_c'].min() == 15
    assert cleaned['temperature_c'].max() == 35