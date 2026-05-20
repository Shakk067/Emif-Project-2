import numpy as np
import pandas as pd


def compute_skewness(series: pd.Series) -> float:
    """
    Manual skewness calculation.
    """
    x = series.dropna()
    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0:
        return np.nan

    return np.mean(((x - mean) / std) ** 3)


def compute_excess_kurtosis(series: pd.Series) -> float:
    """
    Manual excess kurtosis calculation.
    Normal distribution has excess kurtosis = 0.
    """
    x = series.dropna()
    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0:
        return np.nan

    kurtosis = np.mean(((x - mean) / std) ** 4)

    return kurtosis - 3


def compute_jarque_bera(series: pd.Series) -> tuple[float, float]:
    """
    Manual Jarque-Bera test.

    JB = T * (S^2 / 6 + K^2 / 24)

    where:
    S = skewness
    K = excess kurtosis

    Under the null of normality, JB follows chi-square with 2 degrees of freedom.
    For chi-square(2), p-value = exp(-JB / 2).
    """
    x = series.dropna()
    t = len(x)

    s = compute_skewness(x)
    k = compute_excess_kurtosis(x)

    jb_stat = t * ((s**2) / 6 + (k**2) / 24)
    jb_pvalue = np.exp(-jb_stat / 2)

    return jb_stat, jb_pvalue


def compute_risk_metrics(returns_df: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
    """
    Compute standard risk metrics for each asset.

    Metrics:
    - annualized mean
    - annualized volatility
    - skewness
    - excess kurtosis
    - historical VaR 5%
    - historical CVaR 5%
    - Jarque-Bera statistic
    - Jarque-Bera p-value
    """

    rows = []

    for col in returns_df.columns:
        series = returns_df[col].dropna()

        var_5 = series.quantile(0.05)
        cvar_5 = series[series <= var_5].mean()

        jb_stat, jb_pvalue = compute_jarque_bera(series)

        rows.append(
            {
                "Asset": col,
                "Mean_ann": series.mean() * annualization,
                "Vol_ann": series.std() * np.sqrt(annualization),
                "Skewness": compute_skewness(series),
                "Excess_Kurtosis": compute_excess_kurtosis(series),
                "VaR_5": var_5,
                "CVaR_5": cvar_5,
                "JB_stat": jb_stat,
                "JB_pvalue": jb_pvalue,
            }
        )

    result = pd.DataFrame(rows)
    result = result.set_index("Asset")

    return result


def compare_pre_post_risk(pre_covid: pd.DataFrame, post_covid: pd.DataFrame) -> pd.DataFrame:
    """
    Compare risk metrics before and after COVID.
    """

    pre_metrics = compute_risk_metrics(pre_covid)
    post_metrics = compute_risk_metrics(post_covid)

    comparison = pd.DataFrame(index=pre_metrics.index)

    comparison["Vol_ann_pre"] = pre_metrics["Vol_ann"]
    comparison["Vol_ann_post"] = post_metrics["Vol_ann"]
    comparison["Vol_change"] = comparison["Vol_ann_post"] - comparison["Vol_ann_pre"]
    comparison["Vol_change_%"] = comparison["Vol_change"] / comparison["Vol_ann_pre"]

    comparison["Skew_pre"] = pre_metrics["Skewness"]
    comparison["Skew_post"] = post_metrics["Skewness"]

    comparison["Excess_Kurtosis_pre"] = pre_metrics["Excess_Kurtosis"]
    comparison["Excess_Kurtosis_post"] = post_metrics["Excess_Kurtosis"]

    comparison["VaR_5_pre"] = pre_metrics["VaR_5"]
    comparison["VaR_5_post"] = post_metrics["VaR_5"]

    comparison["CVaR_5_pre"] = pre_metrics["CVaR_5"]
    comparison["CVaR_5_post"] = post_metrics["CVaR_5"]

    return comparison