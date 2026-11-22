
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM


def train_hmm(data, min_states=2, max_states=6):
    """Train Gaussian HMM models and select the best one."""

    best_model = None
    best_bic = float("inf")
    best_states = 0

    for states in range(min_states, max_states + 1):

        model = GaussianHMM(
            n_components=states,
            covariance_type="diag",
            n_iter=200,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        log_likelihood = model.score(data)

        n_samples, n_features = data.shape

        parameters = (
            states - 1
            + states * (states - 1)
            + 2 * states * n_features
        )

        bic = (
            -2 * log_likelihood
            + parameters * np.log(n_samples)
        )

        print(f"States: {states}, BIC: {bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_states = states

    print("Best number of hidden states:", best_states)

    return best_model
