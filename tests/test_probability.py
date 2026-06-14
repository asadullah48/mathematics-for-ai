"""
Tests for Probability module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_utils.probability import Probability


class TestDiscreteDistributions:
    """Test discrete probability distributions."""

    def test_bernoulli_pmf(self):
        assert np.isclose(Probability.bernoulli_pmf(1, 0.7), 0.7)
        assert np.isclose(Probability.bernoulli_pmf(0, 0.7), 0.3)
        assert Probability.bernoulli_pmf(2, 0.7) == 0.0

    def test_binomial_pmf_sums_to_one(self):
        n, p = 10, 0.4
        total = sum(Probability.binomial_pmf(k, n, p) for k in range(n + 1))
        assert np.isclose(total, 1.0)

    def test_binomial_pmf_known_value(self):
        # P(X=2 | n=4, p=0.5) = C(4,2) * 0.5^4 = 6/16 = 0.375
        result = Probability.binomial_pmf(2, 4, 0.5)
        assert np.isclose(result, 0.375)

    def test_poisson_pmf_sums_near_one(self):
        lam = 3.0
        total = sum(Probability.poisson_pmf(k, lam) for k in range(100))
        assert np.isclose(total, 1.0, atol=1e-6)

    def test_geometric_pmf_positive_and_decreasing(self):
        p = 0.3
        probs = [Probability.geometric_pmf(k, p) for k in range(1, 10)]
        assert all(pr > 0 for pr in probs)
        assert probs == sorted(probs, reverse=True)


class TestContinuousDistributions:
    """Test continuous probability distributions."""

    def test_gaussian_pdf_peak_at_mean(self):
        mu, sigma = 2.0, 1.5
        at_mean = Probability.gaussian_pdf(mu, mu, sigma)
        away = Probability.gaussian_pdf(mu + 1, mu, sigma)
        assert at_mean > away

    def test_gaussian_cdf_symmetry(self):
        mu, sigma = 0.0, 1.0
        for x in [0.5, 1.0, 2.0]:
            assert np.isclose(
                Probability.gaussian_cdf(mu - x, mu, sigma) +
                Probability.gaussian_cdf(mu + x, mu, sigma),
                1.0, atol=1e-8,
            )

    def test_uniform_pdf_constant_inside(self):
        a, b = 2.0, 5.0
        for x in [2.5, 3.0, 4.9]:
            assert np.isclose(Probability.uniform_pdf(x, a, b), 1.0 / (b - a))
        assert Probability.uniform_pdf(1.9, a, b) == 0.0
        assert Probability.uniform_pdf(5.1, a, b) == 0.0

    def test_uniform_cdf_endpoints(self):
        a, b = 0.0, 4.0
        assert Probability.uniform_cdf(-1.0, a, b) == 0.0
        assert Probability.uniform_cdf(5.0, a, b) == 1.0
        assert np.isclose(Probability.uniform_cdf(2.0, a, b), 0.5)

    def test_exponential_pdf_non_negative(self):
        lam = 2.0
        for x in [0.0, 0.5, 1.0, 2.0]:
            assert Probability.exponential_pdf(x, lam) >= 0
        assert Probability.exponential_pdf(-1.0, lam) == 0.0

    def test_multivariate_gaussian_pdf_positive(self):
        mu = np.array([0.0, 0.0])
        Sigma = np.eye(2)
        x = np.array([0.5, 0.5])
        val = Probability.multivariate_gaussian_pdf(x, mu, Sigma)
        assert val > 0


class TestBayesianInference:
    """Test Bayes' theorem."""

    def test_bayes_theorem_medical_test(self):
        """Classic example: sensitivity 99%, specificity 95%, prevalence 1%."""
        p_disease = 0.01
        p_pos_given_disease = 0.99
        p_pos_given_no_disease = 0.05
        p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * (1 - p_disease)
        posterior = Probability.bayes_theorem(p_disease, p_pos_given_disease, p_pos)
        assert 0.15 < posterior < 0.20

    def test_bayes_theorem_full_matches_manual(self):
        p_a, p_b_given_a, p_b_given_not_a = 0.3, 0.8, 0.2
        result = Probability.bayes_theorem_full(p_a, p_b_given_a, p_b_given_not_a)
        p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
        expected = (p_b_given_a * p_a) / p_b
        assert np.isclose(result, expected)

    def test_bayes_zero_denominator_raises(self):
        with pytest.raises(ValueError):
            Probability.bayes_theorem(0.5, 0.8, 0.0)


class TestExpectationVariance:
    """Test expectation and variance."""

    def test_expectation_fair_die(self):
        values = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        probs = np.ones(6) / 6
        assert np.isclose(Probability.expectation(values, probs), 3.5)

    def test_variance_fair_die(self):
        values = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        probs = np.ones(6) / 6
        var = Probability.variance(values, probs)
        assert np.isclose(var, 35 / 12)

    def test_correlation_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 1.0
        corr = Probability.correlation(x, y)
        assert np.isclose(corr, 1.0)

    def test_correlation_in_range(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        corr = Probability.correlation(x, y)
        assert -1.0 <= corr <= 1.0


class TestMLE:
    """Test Maximum Likelihood Estimation."""

    def test_mle_gaussian_recovers_parameters(self):
        rng = np.random.default_rng(42)
        true_mu, true_sigma = 3.0, 2.0
        data = rng.normal(true_mu, true_sigma, size=10000)
        mu_hat, sigma_hat = Probability.mle_gaussian(data)
        assert np.isclose(mu_hat, true_mu, atol=0.1)
        assert np.isclose(sigma_hat, true_sigma, atol=0.1)

    def test_mle_bernoulli(self):
        rng = np.random.default_rng(1)
        data = rng.binomial(1, 0.7, size=10000)
        p_hat = Probability.mle_bernoulli(data)
        assert np.isclose(p_hat, 0.7, atol=0.02)

    def test_mle_poisson(self):
        rng = np.random.default_rng(2)
        data = rng.poisson(4.5, size=10000)
        lam_hat = Probability.mle_poisson(data)
        assert np.isclose(lam_hat, 4.5, atol=0.1)

    def test_mle_exponential(self):
        rng = np.random.default_rng(3)
        lam = 2.0
        data = rng.exponential(1.0 / lam, size=10000)
        lam_hat = Probability.mle_exponential(data)
        assert np.isclose(lam_hat, lam, atol=0.1)


class TestInformationTheory:
    """Test entropy and divergence."""

    def test_entropy_uniform_maximum(self):
        uniform = np.ones(4) / 4
        peaked = np.array([0.97, 0.01, 0.01, 0.01])
        assert Probability.entropy(uniform) > Probability.entropy(peaked)

    def test_entropy_deterministic_zero(self):
        p = np.array([1.0, 0.0, 0.0])
        assert np.isclose(Probability.entropy(p), 0.0)

    def test_kl_divergence_self_zero(self):
        p = np.array([0.2, 0.5, 0.3])
        assert np.isclose(Probability.kl_divergence(p, p), 0.0, atol=1e-8)

    def test_kl_divergence_non_negative(self):
        p = np.array([0.4, 0.3, 0.3])
        q = np.array([0.1, 0.6, 0.3])
        assert Probability.kl_divergence(p, q) >= 0

    def test_cross_entropy_ge_entropy(self):
        """H(p, q) >= H(p) for all p, q."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        assert Probability.cross_entropy(p, q) >= Probability.entropy(p) - 1e-10


class TestHypothesisTesting:
    """Test hypothesis testing — especially the chi_squared_test bug fix."""

    def test_z_test_returns_scalars(self):
        sample = np.random.randn(50) + 0.5
        z, p = Probability.z_test(sample, pop_mean=0.0, pop_std=1.0)
        assert isinstance(z, (float, np.floating))
        assert 0.0 <= p <= 1.0

    def test_t_test_null_mean_gives_p_near_one(self):
        """When sample mean equals pop mean, t-stat ≈ 0, p-value ≈ 1."""
        rng = np.random.default_rng(99)
        sample = rng.normal(5.0, 1.0, size=200)
        t, p = Probability.t_test_one_sample(sample, pop_mean=np.mean(sample))
        assert np.isclose(t, 0.0, atol=1e-10)
        assert np.isclose(p, 1.0, atol=0.01)

    def test_chi_squared_test_returns_numeric_scalars(self):
        """
        Regression test for the name-shadowing bug:
        'from scipy.stats import chi2' must not overwrite the computed chi2 statistic.
        The function must return (float, float), not (scipy-distribution, float).
        """
        observed = np.array([10.0, 20.0, 30.0])
        expected = np.array([20.0, 20.0, 20.0])
        chi2_stat, p_value = Probability.chi_squared_test(observed, expected)
        assert isinstance(chi2_stat, (float, np.floating)), (
            f"chi2_stat is {type(chi2_stat)}, expected float — "
            "likely the scipy chi2 object was returned instead of the statistic"
        )
        assert isinstance(p_value, (float, np.floating))
        assert chi2_stat > 0
        assert 0.0 <= p_value <= 1.0

    def test_chi_squared_test_uniform_fails_to_reject(self):
        """Perfectly uniform observations → large p-value (fail to reject H0)."""
        observed = np.array([25.0, 25.0, 25.0, 25.0])
        expected = np.array([25.0, 25.0, 25.0, 25.0])
        _, p_value = Probability.chi_squared_test(observed, expected)
        assert p_value > 0.05


class TestSamplingMethods:
    """Test MCMC samplers."""

    def test_metropolis_hastings_shape(self):
        target_log_pdf = lambda x: -0.5 * x ** 2
        samples = Probability.metropolis_hastings(
            target_log_pdf, proposal_std=0.5, n_samples=200, seed=42)
        assert samples.shape == (200,)

    def test_metropolis_hastings_approximate_normal(self):
        """Samples targeting N(0,1) log-pdf should be approximately standard normal."""
        target_log_pdf = lambda x: -0.5 * x ** 2
        samples = Probability.metropolis_hastings(
            target_log_pdf, proposal_std=1.0, n_samples=5000, seed=0)
        samples = samples[500:]  # discard burn-in
        assert abs(np.mean(samples)) < 0.15
        assert 0.7 < np.std(samples) < 1.4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
