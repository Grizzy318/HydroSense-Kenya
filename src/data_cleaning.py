import pandas as pd

def load_dataset(filepath):
    """Load a CSV dataset."""
    return pd.read_csv(filepath)

def remove_duplicates(df):
    return df.drop_duplicates()

def fill_missing_values(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df

def validate_ranges(df, column, minimum=None, maximum=None):
    if minimum is not None:
        df = df[df[column] >= minimum]
    if maximum is not None:
        df = df[df[column] <= maximum]
    return df

def clean_weather_data(filepath):
    df = load_dataset(filepath)
    df = remove_duplicates(df)
    df = fill_missing_values(df)
    if "temperature_c" in df.columns:
        df = validate_ranges(df, "temperature_c", -10, 60)
    return df

def clean_soil_data(filepath):
    df = load_dataset(filepath)
    df = remove_duplicates(df)
    df = fill_missing_values(df)
    if "soil_moisture_pct" in df.columns:
        df = validate_ranges(df, "soil_moisture_pct", 0, 100)
    return df

def clean_crop_data(filepath):
    df = load_dataset(filepath)
    return remove_duplicates(df)
