from pathlib import Path

from src.data_loader import load_raw_data
from src.transformations import prepare_returns, split_pre_post_covid
from src.risk_metrics import compare_pre_post_risk
from src.plots import (
    plot_volatility_change,
    plot_block_correlation_change,
    plot_correlation_difference_heatmap,
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

if __name__ == "__main__":
    main()
