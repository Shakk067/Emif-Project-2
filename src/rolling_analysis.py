import pandas as pd


def compute_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 252,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling annualized volatility.
    """

    rolling_vol = returns_df.rolling(window=window).std() * (annualization ** 0.5)

    return rolling_vol


def compute_rolling_average_correlation(
    returns_df: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """
    Compute rolling average pairwise correlation across all assets.
    """

    values = []
    dates = []

    for i in range(window, len(returns_df) + 1):
        window_data = returns_df.iloc[i - window:i]
        corr = window_data.corr()

        n = corr.shape[0]
        avg_corr = (corr.values.sum() - n) / (n * (n - 1))

        values.append(avg_corr)
        dates.append(returns_df.index[i - 1])

    return pd.Series(values, index=dates, name="Rolling_average_correlation")