from src.data_loader import load_raw_data
from src.transformations import prepare_returns, split_pre_post_covid


def main():
    raw_df = load_raw_data("data/Data.xlsx")

    print("Raw data loaded successfully.")
    print("Raw shape:", raw_df.shape)
    print("Start date:", raw_df.index.min())
    print("End date:", raw_df.index.max())

    returns_df = prepare_returns(raw_df)

    print("\nReturns data prepared successfully.")
    print("Returns shape:", returns_df.shape)
    print("Returns start date:", returns_df.index.min())
    print("Returns end date:", returns_df.index.max())

    pre_covid, post_covid = split_pre_post_covid(returns_df)

    print("\nSample split:")
    print("Pre-COVID shape:", pre_covid.shape)
    print("Pre-COVID start:", pre_covid.index.min())
    print("Pre-COVID end:", pre_covid.index.max())

    print("\nPost-COVID shape:", post_covid.shape)
    print("Post-COVID start:", post_covid.index.min())
    print("Post-COVID end:", post_covid.index.max())

    print("\nFirst rows of transformed data:")
    print(returns_df.head())

    print("\nSummary statistics:")
    print(returns_df.describe())


if __name__ == "__main__":
    main()