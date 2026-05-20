from pathlib import Path

from src.data_loader import load_raw_data
from src.transformations import prepare_returns, split_pre_post_covid
from src.risk_metrics import compare_pre_post_risk
from src.plots import plot_volatility_change


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

    risk_comparison = compare_pre_post_risk(pre_covid, post_covid)

    print("\nRisk comparison: pre-COVID vs post-COVID")
    print(risk_comparison.round(4))

    # Save table
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    risk_comparison.to_csv(table_dir / "risk_comparison_pre_post.csv")

    print("\nSaved table to results/tables/risk_comparison_pre_post.csv")

    # Save figure
    plot_volatility_change(
        risk_comparison,
        output_path="results/figures/volatility_change_pre_post.png",
    )

    print("Saved figure to results/figures/volatility_change_pre_post.png")


if __name__ == "__main__":
    main()