from pathlib import Path
from src.pca_analysis import compare_pre_post_pca
from src.rolling_analysis import (
    compute_rolling_volatility,
    compute_rolling_average_correlation,
)
from src.data_loader import load_raw_data
from src.transformations import prepare_returns, split_pre_post_covid
from src.risk_metrics import compare_pre_post_risk
from src.plots import (
    plot_volatility_change,
    plot_block_correlation_change,
    plot_correlation_difference_heatmap,
    plot_pca_explained_variance,
    plot_rolling_volatility,
    plot_rolling_average_correlation,
)

from src.correlation_analysis import (
    compare_pre_post_correlations,
    summarize_correlation_change,
    summarize_block_correlations,
)


def main():
    raw_df = load_raw_data("data/Data.xlsx")
    returns_df = prepare_returns(raw_df)
    pre_covid, post_covid = split_pre_post_covid(returns_df)

    print("Returns data prepared successfully.")
    print("Returns shape:", returns_df.shape)
    print("Returns start date:", returns_df.index.min())
    print("Returns end date:", returns_df.index.max())

    print("\nSample split:")
    print("Pre-COVID shape:", pre_covid.shape)
    print("Post-COVID shape:", post_covid.shape)

    table_dir = Path("results/tables")
    figure_dir = Path("results/figures")
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. Risk metrics
    risk_comparison = compare_pre_post_risk(pre_covid, post_covid)

    print("\nRisk comparison: pre-COVID vs post-COVID")
    print(risk_comparison.round(4))

    risk_comparison.to_csv(table_dir / "risk_comparison_pre_post.csv")

    plot_volatility_change(
        risk_comparison,
        output_path=figure_dir / "volatility_change_pre_post.png",
    )

    # 2. Correlation analysis
    corr_pre, corr_post, corr_diff = compare_pre_post_correlations(
        pre_covid,
        post_covid,
    )

    corr_summary = summarize_correlation_change(corr_pre, corr_post)
    block_corr_summary = summarize_block_correlations(corr_pre, corr_post)

    print("\nAverage correlation summary:")
    print(corr_summary.round(4))

    print("\nBlock correlation summary:")
    print(block_corr_summary.round(4))

    corr_pre.to_csv(table_dir / "correlation_matrix_pre_covid.csv")
    corr_post.to_csv(table_dir / "correlation_matrix_post_covid.csv")
    corr_diff.to_csv(table_dir / "correlation_matrix_difference.csv")
    corr_summary.to_csv(table_dir / "correlation_summary.csv", index=False)
    block_corr_summary.to_csv(table_dir / "block_correlation_summary.csv")

    print("\nSaved correlation tables to results/tables/")
    plot_block_correlation_change(
        block_corr_summary,
        output_path=figure_dir / "block_correlation_change.png",
    )

    plot_correlation_difference_heatmap(
        corr_diff,
        output_path=figure_dir / "correlation_difference_heatmap.png",
    )

    print("Saved correlation figures to results/figures/")
        # 3. PCA analysis
    explained_pre, explained_post, pca_comparison, loadings_pre, loadings_post = (
        compare_pre_post_pca(pre_covid, post_covid)
    )

    print("\nPCA explained variance comparison:")
    print(pca_comparison.head(5).round(4))

    print("\nPCA first component loadings pre-COVID:")
    print(loadings_pre["PC1"].sort_values(ascending=False).round(4))

    print("\nPCA first component loadings post-COVID:")
    print(loadings_post["PC1"].sort_values(ascending=False).round(4))

    explained_pre.to_csv(table_dir / "pca_explained_variance_pre.csv")
    explained_post.to_csv(table_dir / "pca_explained_variance_post.csv")
    pca_comparison.to_csv(table_dir / "pca_explained_variance_comparison.csv")
    loadings_pre.to_csv(table_dir / "pca_loadings_pre.csv")
    loadings_post.to_csv(table_dir / "pca_loadings_post.csv")

    plot_pca_explained_variance(
        pca_comparison,
        output_path=figure_dir / "pca_explained_variance_pre_post.png",
    )

    print("\nSaved PCA tables and figures.")
        # 4. Rolling analysis
    rolling_vol = compute_rolling_volatility(returns_df, window=252)
    rolling_corr = compute_rolling_average_correlation(returns_df, window=252)

    rolling_vol.to_csv(table_dir / "rolling_volatility.csv")
    rolling_corr.to_csv(table_dir / "rolling_average_correlation.csv")

    selected_assets = [
        "S&P500",
        "US IG Bonds",
        "US HY Bonds",
        "Oil futures",
        "Gold",
    ]

    plot_rolling_volatility(
        rolling_vol,
        selected_assets,
        output_path=figure_dir / "rolling_volatility_selected_assets.png",
    )

    plot_rolling_average_correlation(
        rolling_corr,
        output_path=figure_dir / "rolling_average_correlation.png",
    )

    print("\nSaved rolling analysis tables and figures.")


if __name__ == "__main__":
    main()
