from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_volatility_change(
    risk_comparison: pd.DataFrame,
    output_path: str = "results/figures/volatility_change_pre_post.png",
) -> None:
    """
    Plot the change in annualized volatility between pre-COVID and post-COVID periods.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = risk_comparison["Vol_change"].sort_values()

    plt.figure(figsize=(12, 7))
    data.plot(kind="barh")

    plt.axvline(0, linewidth=1)
    plt.title("Change in Annualized Risk After COVID-19")
    plt.xlabel("Post-COVID annualized volatility minus pre-COVID annualized volatility")
    plt.ylabel("Asset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_block_correlation_change(
    block_corr_summary: pd.DataFrame,
    output_path: str = "results/figures/block_correlation_change.png",
) -> None:
    """
    Plot changes in block correlations between pre-COVID and post-COVID periods.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = block_corr_summary["Change"].sort_values()

    plt.figure(figsize=(12, 7))
    data.plot(kind="barh")

    plt.axvline(0, linewidth=1)
    plt.title("Change in Asset-Class Correlations After COVID-19")
    plt.xlabel("Post-COVID correlation minus pre-COVID correlation")
    plt.ylabel("Correlation block")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_difference_heatmap(
    corr_diff: pd.DataFrame,
    output_path: str = "results/figures/correlation_difference_heatmap.png",
) -> None:
    """
    Plot a heatmap of post-COVID minus pre-COVID correlations.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))

    plt.imshow(corr_diff, aspect="auto")
    plt.colorbar(label="Correlation difference")

    plt.xticks(range(len(corr_diff.columns)), corr_diff.columns, rotation=90)
    plt.yticks(range(len(corr_diff.index)), corr_diff.index)

    plt.title("Correlation Matrix Difference: Post-COVID minus Pre-COVID")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
     
def plot_pca_explained_variance(
    pca_comparison: pd.DataFrame,
    output_path: str = "results/figures/pca_explained_variance_pre_post.png",
    n_components: int = 5,
) -> None:
    """
    Plot explained variance ratios for the first PCA components,
    before and after COVID.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pca_comparison.iloc[:n_components][["Explained_pre", "Explained_post"]]

    plt.figure(figsize=(10, 6))
    data.plot(kind="bar")

    plt.title("PCA: Explained Variance Before and After COVID-19")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  

def plot_rolling_volatility(
    rolling_vol: pd.DataFrame,
    assets: list[str],
    output_path: str = "results/figures/rolling_volatility_selected_assets.png",
) -> None:
    """
    Plot rolling annualized volatility for selected assets.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    for asset in assets:
        plt.plot(rolling_vol.index, rolling_vol[asset], label=asset)

    plt.axvline(pd.to_datetime("2020-03-11"), linewidth=1, linestyle="--")
    plt.title("Rolling Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rolling_average_correlation(
    rolling_corr: pd.Series,
    output_path: str = "results/figures/rolling_average_correlation.png",
) -> None:
    """
    Plot rolling average pairwise correlation across all assets.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(rolling_corr.index, rolling_corr.values)

    plt.axvline(pd.to_datetime("2020-03-11"), linewidth=1, linestyle="--")
    plt.title("Rolling Average Cross-Asset Correlation")
    plt.xlabel("Date")
    plt.ylabel("Average pairwise correlation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()   