import numpy as np


# Mock ArtifactStore
class MockStore:
    def __init__(self, data):
        self.data = data

    def require(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)


# Copy-paste the Modified PcaStep (simplified context) to test logic in isolation
# OR import it if possible. Importing is better.
try:
    from modssc.preprocess.steps.core.pca import PcaStep
except ImportError:
    # If python path issues, add src
    import os
    import sys

    sys.path.append(os.path.abspath("src"))
    from modssc.preprocess.steps.core.pca import PcaStep


def test_pca_chunking():
    print("Testing PCA Chunking...")
    # Generate synthetic data
    n_samples = 20000  # Enough to trigger chunking if batch_size is 8192
    n_features = 50
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))

    store = MockStore({"features.X": X})

    # Fit
    step = PcaStep(n_components=10, center=True)
    fit_indices = np.arange(n_samples)  # Fit on all
    step.fit(store, fit_indices=fit_indices, rng=rng)

    # Transform (this uses the NEW chunked logic)
    res_chunked = step.transform(store, rng=rng)
    Z_chunked = res_chunked["features.X"]

    # Manual Transform (No chunking logic) for verification
    mean = step.mean_
    components = step.components_
    X_centered = X - mean
    Z_expected = X_centered @ components.T

    # Comparison
    # Note: The new logic casts output to float32, so we compare with that tolerance
    max_diff = np.abs(Z_chunked - Z_expected.astype(np.float32)).max()
    print(f"Max difference between Chunked and Full: {max_diff}")

    if max_diff < 1e-5:
        print("PCA Chunking Test PASSED")
    else:
        print("PCA Chunking Test FAILED")
        exit(1)


if __name__ == "__main__":
    test_pca_chunking()
