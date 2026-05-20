import pandas as pd

from src.transformations import split_pre_post_covid
from src.risk_metrics import compare_pre_post_risk
from src.correlation_analysis import (
    compare_pre_post_correlations,
    summarize_correlation_change,
)
from src.pca_analysis import compare_pre_post_pca


def run_breakpoint_robustness(
    returns_df: pd.DataFrame,
    breakpoints: list[str],
) -> pd.DataFrame:
    """
    Run key results across alternative COVID breakpoints.
    """

    rows = []

    for breakpoint in breakpoints:
        pre, post = split_pre_post_covid(returns_df, breakpoint=breakpoint)

        risk_comparison = compare_pre_post_risk(pre, post)

        corr_pre, corr_post, _ = compare_pre_post_correlations(pre, post)
        corr_summary = summarize_correlation_change(corr_pre, corr_post)

        _, _, pca_comparison, _, _ = compare_pre_post_pca(pre, post)

        rows.append(
            {
                "Breakpoint": breakpoint,
                "Pre_obs": len(pre),
                "Post_obs": len(post),
                "Average_vol_change": risk_comparison["Vol_change"].mean(),
                "Average_corr_pre": corr_summary.loc[0, "Average_correlation_pre"],
                "Average_corr_post": corr_summary.loc[0, "Average_correlation_post"],
                "Average_corr_change": corr_summary.loc[0, "Change"],
                "PC1_pre": pca_comparison.loc["PC1", "Explained_pre"],
                "PC1_post": pca_comparison.loc["PC1", "Explained_post"],
                "PC1_change": pca_comparison.loc["PC1", "Change"],
                "PC1_PC2_pre": pca_comparison.loc[["PC1", "PC2"], "Explained_pre"].sum(),
                "PC1_PC2_post": pca_comparison.loc[["PC1", "PC2"], "Explained_post"].sum(),
                "PC1_PC2_change": pca_comparison.loc[["PC1", "PC2"], "Change"].sum(),
            }
        )

    return pd.DataFrame(rows).set_index("Breakpoint")


def run_no_oil_robustness(
    returns_df: pd.DataFrame,
    breakpoint: str = "2020-03-11",
) -> dict[str, pd.DataFrame]:
    """
    Re-run key results after excluding Oil futures.
    """

    returns_no_oil = returns_df.drop(columns=["Oil futures"])

    pre, post = split_pre_post_covid(returns_no_oil, breakpoint=breakpoint)

    risk_comparison = compare_pre_post_risk(pre, post)

    corr_pre, corr_post, corr_diff = compare_pre_post_correlations(pre, post)
    corr_summary = summarize_correlation_change(corr_pre, corr_post)

    explained_pre, explained_post, pca_comparison, loadings_pre, loadings_post = (
        compare_pre_post_pca(pre, post)
    )

    return {
        "risk_comparison_no_oil": risk_comparison,
        "correlation_summary_no_oil": corr_summary,
        "correlation_difference_no_oil": corr_diff,
        "pca_comparison_no_oil": pca_comparison,
        "pca_loadings_pre_no_oil": loadings_pre,
        "pca_loadings_post_no_oil": loadings_post,
    }