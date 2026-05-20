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