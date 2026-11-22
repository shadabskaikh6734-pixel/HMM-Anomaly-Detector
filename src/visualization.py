
import matplotlib.pyplot as plt
import numpy as np


def plot_anomalies(
    test_data,
    anomalies,
    window_size=50,
    step=5
):
    """Plot sensor readings and detected anomaly regions."""

    readings = test_data.flatten()
    time = np.arange(len(readings))

    plt.figure(figsize=(14, 6))

    plt.plot(
        time,
        readings,
        label="Sensor Readings",
        color="blue"
    )

    # Highlight anomaly regions
    for i, is_anomaly in enumerate(anomalies):

        if is_anomaly:
            start = i * step
            end = min(
                start + window_size,
                len(readings)
            )

            plt.axvspan(
                start,
                end,
                color="red",
                alpha=0.3
            )

    plt.title("HMM Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/anomaly_detection_result.png"
    )

    plt.show()
