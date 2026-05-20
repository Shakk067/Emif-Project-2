from src.data_loader import load_raw_data


def main():
    df = load_raw_data("data/Data.xlsx")

    print("Data loaded successfully.")
    print("Shape:", df.shape)
    print("Start date:", df.index.min())
    print("End date:", df.index.max())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
    