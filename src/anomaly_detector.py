
import numpy as np


def create_windows(data, window_size=50, step=5):
    """Create sliding windows from sensor data."""

    windows = []

    for start in range(0, len(data) - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end])

    return windows


def detect_anomalies(
    model,
    train_data,
    test_data,
    window_size=50,
    step=5,
    percentile=1
):
    """Detect anomalies using HMM log-likelihood."""

    train_windows = create_windows(
        train_data, window_size, step
    )

    test_windows = create_windows(
        test_data, window_size, step
    )

    # Calculate normal training scores
    train_scores = []

    for window in train_windows:
        score = model.score(window) / len(window)
        train_scores.append(score)

    # Calculate test scores
    test_scores = []

    for window in test_windows:
        score = model.score(window) / len(window)
        test_scores.append(score)

    # Set threshold from normal training data
    threshold = np.percentile(
        train_scores, percentile
    )

    # Flag anomalous windows
    anomalies = np.array(test_scores) < threshold

    print("Anomaly Threshold:", threshold)
    print("Detected Anomalies:", anomalies.sum())

    return test_scores, anomalies, threshold
