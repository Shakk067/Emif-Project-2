import pandas as pd


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix of asset returns.
    """
    return returns_df.corr()


def compare_pre_post_correlations(
    pre_covid: pd.DataFrame,
    post_covid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pre-COVID, post-COVID and difference correlation matrices.

    Difference = post-COVID correlation - pre-COVID correlation.
    """

    corr_pre = compute_correlation_matrix(pre_covid)
    corr_post = compute_correlation_matrix(post_covid)
    corr_diff = corr_post - corr_pre

    return corr_pre, corr_post, corr_diff


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    Compute the average pairwise correlation, excluding the diagonal.
    """

    n = corr_matrix.shape[0]

    total_sum = corr_matrix.values.sum()
    diagonal_sum = n

    average_corr = (total_sum - diagonal_sum) / (n * (n - 1))

    return average_corr


def summarize_correlation_change(
    corr_pre: pd.DataFrame,
    corr_post: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the average pairwise correlation before and after COVID.
    """

    avg_pre = average_pairwise_correlation(corr_pre)
    avg_post = average_pairwise_correlation(corr_post)

    summary = pd.DataFrame(
        {
            "Average_correlation_pre": [avg_pre],
            "Average_correlation_post": [avg_post],
            "Change": [avg_post - avg_pre],
        }
    )

    return summary


def average_block_correlation(
    corr_matrix: pd.DataFrame,
    assets_1: list[str],
    assets_2: list[str] | None = None,
) -> float:
    """
    Compute average correlation within or between asset blocks.

    If assets_2 is None, it computes within-block average correlation.
    If assets_2 is provided, it computes between-block average correlation.
    """

    if assets_2 is None:
        sub_corr = corr_matrix.loc[assets_1, assets_1]

        n = len(assets_1)
        if n <= 1:
            return float("nan")

        total_sum = sub_corr.values.sum()
        diagonal_sum = n

        return (total_sum - diagonal_sum) / (n * (n - 1))

    sub_corr = corr_matrix.loc[assets_1, assets_2]
    return sub_corr.values.mean()


def summarize_block_correlations(
    corr_pre: pd.DataFrame,
    corr_post: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize correlation changes by asset class blocks.
    """

    equities = ["S&P500", "Eurostoxx 50", "Hang Seng", "MSCI EM", "SMI"]
    yields = ["US T 10-year Yield", "German Gov 10-year yield"]
    commodities = ["Oil futures", "Gold"]
    fx = ["EURUSD", "USDJPY", "USDCHF"]
    credit = ["US IG Bonds", "US HY Bonds"]

    blocks = {
        "Equities within": (equities, None),
        "Yields within": (yields, None),
        "Commodities within": (commodities, None),
        "FX within": (fx, None),
        "Credit within": (credit, None),
        "Equities vs Yields": (equities, yields),
        "Equities vs Credit": (equities, credit),
        "Equities vs Commodities": (equities, commodities),
        "Equities vs FX": (equities, fx),
        "Credit vs Yields": (credit, yields),
    }

    rows = []

    for name, (assets_1, assets_2) in blocks.items():
        pre_value = average_block_correlation(corr_pre, assets_1, assets_2)
        post_value = average_block_correlation(corr_post, assets_1, assets_2)

        rows.append(
            {
                "Block": name,
                "Correlation_pre": pre_value,
                "Correlation_post": post_value,
                "Change": post_value - pre_value,
            }
        )

    return pd.DataFrame(rows).set_index("Block")