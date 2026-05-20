import numpy as np
import pandas as pd


LOG_RETURN_COLUMNS = [
    "S&P500",
    "Eurostoxx 50",
    "Hang Seng",
    "MSCI EM",
    "SMI",
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

SPECIAL_COLUMNS = [
    "Oil futures",
]


def prepare_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw market data into stationary financial series.

    - Price/index/bond/FX series are transformed into log returns.
    - Government bond yields are transformed into daily changes in basis points.
    - Oil futures are transformed with an asinh difference because oil prices became
      negative in April 2020, making log returns impossible.
    """

    df = raw_df.copy()

    # Clean index name
    df.index.name = "Date"

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    transformed = pd.DataFrame(index=df.index)

    # Standard log returns
    for col in LOG_RETURN_COLUMNS:
        transformed[col] = np.log(df[col] / df[col].shift(1))

    # Yield changes in basis points
    # If yields are quoted in percentage points, e.g. 4.25, then diff * 100 = basis points
    for col in YIELD_COLUMNS:
        transformed[col] = df[col].diff() * 100

    # Oil futures special treatment
    # asinh(x) behaves like log(x) for large positive values but also works for zero/negative prices
    transformed["Oil futures"] = np.arcsinh(df["Oil futures"]) - np.arcsinh(df["Oil futures"].shift(1))

    # Reorder columns as in the original dataset
    transformed = transformed[df.columns]

    # Drop missing values only after all transformations
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
