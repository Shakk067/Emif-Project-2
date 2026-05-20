from pathlib import Path
import pandas as pd


def load_raw_data(path: str | Path = "data/Data.xlsx") -> pd.DataFrame:
    """
    Load the raw Excel file provided for the EMiF project.
    The function assumes the first column is a date column.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )

    # Detect date column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()

    return df
