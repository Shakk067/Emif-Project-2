import numpy as np
import pandas as pd


def standardize_data(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each asset return series:
    z = (x - mean) / standard deviation

    PCA should be done on standardized data so that high-volatility assets
    do not mechanically dominate the first components.
    """

    return (returns_df - returns_df.mean()) / returns_df.std()


def compute_pca(returns_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute PCA manually using the eigen-decomposition of the correlation matrix.

    Returns:
    - explained_variance table
    - loadings table
    """

    standardized = standardize_data(returns_df)

    corr_matrix = standardized.corr()

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort eigenvalues from largest to smallest
    sorted_indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    explained_ratio = eigenvalues / eigenvalues.sum()
    cumulative_ratio = np.cumsum(explained_ratio)

    component_names = [f"PC{i+1}" for i in range(len(eigenvalues))]

    explained_variance = pd.DataFrame(
        {
            "Eigenvalue": eigenvalues,
            "Explained_variance_ratio": explained_ratio,
            "Cumulative_explained_variance": cumulative_ratio,
        },
        index=component_names,
    )

    loadings = pd.DataFrame(
        eigenvectors,
        index=returns_df.columns,
        columns=component_names,
    )

    return explained_variance, loadings


def compare_pre_post_pca(
    pre_covid: pd.DataFrame,
    post_covid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare PCA results before and after COVID.

    Returns:
    - pre explained variance
    - post explained variance
    - comparison table
    - pre loadings
    - post loadings
    """

    explained_pre, loadings_pre = compute_pca(pre_covid)
    explained_post, loadings_post = compute_pca(post_covid)

    comparison = pd.DataFrame(
        {
            "Explained_pre": explained_pre["Explained_variance_ratio"],
            "Explained_post": explained_post["Explained_variance_ratio"],
            "Change": explained_post["Explained_variance_ratio"]
            - explained_pre["Explained_variance_ratio"],
        }
    )

    return explained_pre, explained_post, comparison, loadings_pre, loadings_post