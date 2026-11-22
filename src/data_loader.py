
import pandas as pd
import numpy as np


def load_sensor_data(file_path, column="sensor_value"):
    """Load sensor readings from a CSV file."""

    data = pd.read_csv(file_path)

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found.")

    sensor_values = data[column].dropna().values

    return sensor_values.reshape(-1, 1)


def generate_demo_data():
    """Generate sample normal and test sensor data."""

    np.random.seed(42)

    normal_data = np.random.normal(20, 2, 1000)

    test_data = np.random.normal(20, 2, 500)

    # Add an abnormal region
    test_data[200:250] = np.random.normal(45, 3, 50)

    return (
        normal_data.reshape(-1, 1),
        test_data.reshape(-1, 1)
    )
