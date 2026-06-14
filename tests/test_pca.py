"""
Tests for PCA (Principal Component Analysis).
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_utils.linear_algebra import PCA


class TestPCAFit:
    """Test PCA fitting."""

    def test_fit_returns_self(self):
        X = np.random.randn(50, 5)
        pca = PCA(n_components=3)
        result = pca.fit(X)
        assert result is pca

    def test_mean_centring(self):
        X = np.random.randn(100, 4) + np.array([1, 2, 3, 4])
        pca = PCA()
        pca.fit(X)
        assert np.allclose(pca.mean_, np.mean(X, axis=0))

    def test_components_shape(self):
        X = np.random.randn(100, 10)
        pca = PCA(n_components=3)
        pca.fit(X)
        assert pca.components_.shape == (3, 10)

    def test_components_orthonormal(self):
        """Principal components must be orthonormal (SVD property)."""
        X = np.random.randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        product = pca.components_ @ pca.components_.T
        assert np.allclose(product, np.eye(5), atol=1e-10)

    def test_explained_variance_positive(self):
        X = np.random.randn(50, 4)
        pca = PCA()
        pca.fit(X)
        assert np.all(pca.explained_variance_ > 0)

    def test_explained_variance_ratio_sums_to_one(self):
        X = np.random.randn(50, 4)
        pca = PCA()
        pca.fit(X)
        assert np.isclose(np.sum(pca.explained_variance_ratio_), 1.0)

    def test_explained_variance_ratio_descending(self):
        X = np.random.randn(100, 6)
        pca = PCA()
        pca.fit(X)
        evr = pca.explained_variance_ratio_
        assert np.all(evr[:-1] >= evr[1:])

    def test_n_components_capped_by_min_dim(self):
        X = np.random.randn(10, 20)  # more features than samples
        pca = PCA()
        pca.fit(X)
        assert pca.n_components_ == 10  # capped at n_samples


class TestPCATransform:
    """Test PCA projection."""

    def test_transform_shape(self):
        X = np.random.randn(80, 10)
        pca = PCA(n_components=3)
        pca.fit(X)
        X_proj = pca.transform(X)
        assert X_proj.shape == (80, 3)

    def test_transform_zero_mean(self):
        """Projected data has near-zero mean because the input was centred."""
        X = np.random.randn(200, 5) + 5
        pca = PCA(n_components=5)
        pca.fit(X)
        X_proj = pca.transform(X)
        assert np.allclose(np.mean(X_proj, axis=0), 0, atol=1e-10)

    def test_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        pca1 = PCA(n_components=2)
        X1 = pca1.fit_transform(X)

        pca2 = PCA(n_components=2)
        pca2.fit(X)
        X2 = pca2.transform(X)

        # Components may differ in sign; compare absolute values
        assert np.allclose(np.abs(X1), np.abs(X2), atol=1e-10)


class TestPCAReconstruction:
    """Test reconstruction and reconstruction error."""

    def test_full_reconstruction_exact(self):
        """Keeping all components should allow perfect reconstruction."""
        X = np.random.randn(40, 4)
        pca = PCA(n_components=4)
        X_proj = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_proj)
        assert np.allclose(X, X_rec, atol=1e-10)

    def test_reconstruction_error_decreases_with_more_components(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 8))
        errors = []
        for k in [1, 2, 4, 8]:
            pca = PCA(n_components=k)
            err = pca.fit(X).reconstruction_error(X)
            errors.append(err)
        assert errors == sorted(errors, reverse=True)

    def test_reconstruction_error_full_is_zero(self):
        X = np.random.randn(30, 3)
        pca = PCA(n_components=3)
        pca.fit(X)
        assert np.isclose(pca.reconstruction_error(X), 0.0, atol=1e-10)


class TestPCAWhitening:
    """Test whitening option."""

    def test_whitened_unit_variance(self):
        """Whitened projected components should have approximately unit variance."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 4))
        pca = PCA(n_components=4, whiten=True)
        X_proj = pca.fit_transform(X)
        var = np.var(X_proj, axis=0, ddof=1)
        assert np.allclose(var, 1.0, atol=0.05)


class TestPCACumulativeVariance:
    """Test cumulative explained variance."""

    def test_cumulative_ends_at_one(self):
        X = np.random.randn(50, 5)
        pca = PCA()
        pca.fit(X)
        cum = pca.cumulative_explained_variance_ratio
        assert np.isclose(cum[-1], 1.0)

    def test_cumulative_monotone_non_decreasing(self):
        X = np.random.randn(50, 5)
        pca = PCA()
        pca.fit(X)
        cum = pca.cumulative_explained_variance_ratio
        assert np.all(np.diff(cum) >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
