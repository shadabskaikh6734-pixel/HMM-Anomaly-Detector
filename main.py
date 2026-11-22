
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_demo_data
from src.hmm_model import train_hmm
from src.anomaly_detector import detect_anomalies
from src.visualization import plot_anomalies


def main():
    print("HMM Anomaly Detection Started")

    # Load sensor data
    train_data, test_data = generate_demo_data()

    # Normalize data
    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Train HMM
    model = train_hmm(train_data)

    # Detect anomalies
    scores, anomalies, threshold = detect_anomalies(
        model,
        train_data,
        test_data
    )

    # Visualize results
    plot_anomalies(
        test_data,
        anomalies
    )

    print("Anomaly Detection Completed")


if __name__ == "__main__":
    main()
