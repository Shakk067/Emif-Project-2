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