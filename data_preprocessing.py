import pandas as pd
import numpy as np


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Saves the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the DataFrame and removes old features based on existing data.

    These new features can help improve the performance of machine learning models.
    """
    # Check if 'is_done' is of object type (string) before mapping
    if df["is_done"].dtype == "object":
        df["is_done"] = df["is_done"].str.strip().map({"done": 1, "cancel": 0})

    # Convert timestamp to datetime and extract time-based features
    df["order_datetime"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
    df["order_hour"] = df["order_datetime"].dt.hour
    df["order_dayofweek"] = df["order_datetime"].dt.dayofweek
    df["is_night"] = df["order_hour"].apply(lambda x: 1 if 0 <= x < 6 else 0)

    # Calculate price metrics, handling division by zero
    df["price_per_meter"] = np.where(
        df["distance_in_meters"] > 0,
        df["price_bid_local"] / df["distance_in_meters"],
        0,
    )
    df["price_per_second"] = np.where(
        df["duration_in_seconds"] > 0,
        df["price_bid_local"] / df["duration_in_seconds"],
        0,
    )

    # Calculate price increase metrics
    df["price_increase_abs"] = df["price_bid_local"] - df["price_start_local"]
    df["price_increase_perc"] = np.where(
        df["price_start_local"] > 0,
        (df["price_increase_abs"] / df["price_start_local"]) * 100,
        0,
    )

    # Calculate driver experience in days
    df["driver_reg_datetime"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")
    df["driver_experience_days"] = (
        df["order_datetime"] - df["driver_reg_datetime"]
    ).dt.days

    # Correctly impute missing values for driver experience
    median_experience = df["driver_experience_days"].median()
    df["driver_experience_days"] = df["driver_experience_days"].fillna(
        median_experience
    )

    # Логарифмирование признака distance_in_meters, чтобы сгладить его распределение
    # Добавляем +1 (псевдосчет), чтобы избежать ошибки логарифмирования нуля
    if "distance_in_meters" in df.columns:
        df["distance_in_meters_log"] = np.log1p(df["distance_in_meters"])

    df.drop(
        columns=["order_datetime", "driver_reg_datetime"],
        inplace=True,
    )
    return df


def remove_leaky_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Removes columns that could leak target information or are identifiers."""
    cols_to_drop = [
        "order_id",
        "user_id",
        "tender_id",
        "tender_timestamp",
        "order_timestamp",
        "driver_reg_date",
        "driver_id",
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df


def handle_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Handles anomalies and outliers in the data."""
    initial_rows = len(df)

    # Remove rows with negative driver experience days
    df = df[df["driver_experience_days"] >= 0]
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with negative driver experience.")

    # Remove trips with unrealistically short distance and duration
    initial_rows_after_exp_filter = len(df)
    df = df[~((df["distance_in_meters"] < 10) & (df["duration_in_seconds"] < 10))]
    rows_removed_short = initial_rows_after_exp_filter - len(df)
    if rows_removed_short > 0:
        print(f"Removed {rows_removed_short} rows with anomalous short trip data.")

    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Main function to load and preprocess the data.
    """
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Error: file {file_path} not found.")
        raise e
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        # Fallback to latin1 if utf-8 fails
        try:
            print("Attempting to read with 'latin1' encoding.")
            df = pd.read_csv(file_path, sep=",", encoding="latin1")
        except Exception as e_inner:
            print(f"Failed to read with 'latin1' as well: {e_inner}")
            raise e_inner

    if "Unnamed: 18" in df.columns:
        df.drop("Unnamed: 18", axis=1, inplace=True)

    df = feature_engineering(df)
    df = remove_leaky_attributes(df)
    df = handle_anomalies(df)

    return df


if __name__ == "__main__":
    try:
        processed_df = preprocess_data("train.csv")
        save_data(processed_df, "data_processed.csv")
        print("\nPreprocessing complete. Data saved to data_processed.csv")
        print("\nExample of cleaned data:")
        print(processed_df.head())
        print(f"\nFinal data size: {processed_df.shape}")
    except Exception as e:
        print(f"An error occurred during the preprocessing pipeline: {e}")
