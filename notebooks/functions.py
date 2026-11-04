import matplotlib.pyplot as plt
import pandas as pd
import kagglehub

def getKaggleData():
    """
    Download the household electricity consumption dataset from Kaggle.
    """
    
    kagglehub.dataset_download("uciml/electric-power-consumption-data-set", output_dir="../data")

    return

def preprocess_data(path):
    """
    Load and preprocess household electricity consumption data.

    Returns:
        DataFrame: Preprocessed data with a datetime index and hourly averages of measurements.
    """
    # Load raw data
    data = pd.read_csv(path, sep=";", low_memory=False)

    # Convert measurement columns to numeric
    for col in data.columns:
        if col not in ["Date", "Time"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Create datetime column
    data["DateTime"] = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    # Forward-fill missing values
    data = data.ffill()

    # Set DateTime as index and resample to daily averages
    data = (
        data
        .set_index("DateTime")
        .resample("D")
        .mean(numeric_only=True)
    )

    return data



def plot_na_heatmap(data):
    
    measurement_columns = data.select_dtypes(include="number").columns
    missing_mask = data[measurement_columns].isna().to_numpy(dtype="uint8").T

    tick_positions = [int(i * (len(data) - 1) / 5) for i in range(6)]
    tick_labels = data["DateTime"].iloc[tick_positions].dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(16, 5))
    image = ax.imshow(
        missing_mask,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(len(measurement_columns)), labels=measurement_columns)
    ax.set_xticks(tick_positions, labels=tick_labels)
    ax.set_xlabel("Date")
    ax.set_title("Missing values across time")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Present", "Missing"])
    fig.tight_layout()
    plt.show()

    return