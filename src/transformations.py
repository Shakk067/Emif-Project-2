import numpy as np
import pandas as pd


PRICE_COLUMNS = [
    "S&P500",
    "Eurostoxx 50",
    "Hang Seng",
    "MSCI EM",
    "SMI",
    "Oil futures",
    "Gold",
    "EURUSD",
    "USDJPY",
    "US IG Bonds",
    "US HY Bonds",
    "USDCHF",
]

YIELD_COLUMNS = [
    "US T 10-year Yield",
    "German Gov 10-year yield",
]


def prepare_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw market data into stationary financial series.

    - Price/index/bond/commodity/FX series are transformed into log returns.
    - Government bond yields are transformed into daily changes in basis points.
    """

    df = raw_df.copy()

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    transformed = pd.DataFrame(index=df.index)

    # Log returns for price-like series
    for col in PRICE_COLUMNS:
        transformed[col] = np.log(df[col] / df[col].shift(1))

    # Yield changes in basis points
    # If yields are in percentage points, e.g. 4.25, then difference * 100 = bps
    for col in YIELD_COLUMNS:
        transformed[col] = df[col].diff() * 100

    # Reorder columns: keep same order as raw file
    transformed = transformed[df.columns]

    # Drop rows with missing values after transformation
    transformed = transformed.dropna(how="any")

    return transformed


def split_pre_post_covid(
    returns_df: pd.DataFrame,
    breakpoint: str = "2020-03-11",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into pre-COVID and post-COVID samples.

    Default breakpoint: 2020-03-11, the date when WHO characterized COVID-19 as a pandemic.
    """

    breakpoint = pd.to_datetime(breakpoint)

    pre_covid = returns_df.loc[returns_df.index < breakpoint].copy()
    post_covid = returns_df.loc[returns_df.index >= breakpoint].copy()

    return pre_covid, post_covid