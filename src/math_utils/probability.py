"""
Probability Module

Probability theory implementations for machine learning and statistics.
Includes distributions, Bayes' theorem, and probabilistic computations.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Callable
from scipy import stats as scipy_stats
from dataclasses import dataclass


@dataclass
class DistributionStats:
    """Statistics of a probability distribution."""
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float


class Probability:
    """
    Comprehensive probability theory implementations.
    
    Features:
    - Common probability distributions (discrete and continuous)
    - Bayes' theorem and conditional probability
    - Expectation and variance computations
    - Maximum Likelihood Estimation (MLE)
    - Maximum A Posteriori (MAP)
    - Sampling methods
    - Law of large numbers demonstrations
    """
    
    # ==================== Basic Probability ====================
    
    @staticmethod
    def factorial(n: int) -> int:
        """Compute factorial n!"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k)."""
        if k > n or k < 0:
            return 0
        return Probability.factorial(n) // (Probability.factorial(k) * Probability.factorial(n - k))
    
    @staticmethod
    def permutation(n: int, k: int) -> int:
        """Compute permutations P(n, k)."""
        return Probability.factorial(n) // Probability.factorial(n - k)
    
    @staticmethod
    def combination(n: int, k: int) -> int:
        """Compute combinations C(n, k)."""
        return Probability.binomial_coefficient(n, k)
    
    # ==================== Discrete Distributions ====================
    
    @staticmethod
    def bernoulli_pmf(k: int, p: float) -> float:
        """
        Bernoulli distribution probability mass function.
        
        Args:
            k: Outcome (0 or 1)
            p: Probability of success
            
        Returns:
            P(X = k)
        """
        if k not in [0, 1]:
            return 0.0
        return p if k == 1 else (1 - p)
    
    @staticmethod
    def bernoulli_sample(p: float, size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Bernoulli distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.binomial(1, p, size)
    
    @staticmethod
    def binomial_pmf(k: int, n: int, p: float) -> float:
        """
        Binomial distribution probability mass function.
        
        Args:
            k: Number of successes
            n: Number of trials
            p: Probability of success
            
        Returns:
            P(X = k)
        """
        if k < 0 or k > n:
            return 0.0
        coeff = Probability.binomial_coefficient(n, k)
        return coeff * (p ** k) * ((1 - p) ** (n - k))
    
    @staticmethod
    def binomial_cdf(k: int, n: int, p: float) -> float:
        """
        Binomial cumulative distribution function.
        
        Returns:
            P(X ≤ k)
        """
        return sum(Probability.binomial_pmf(i, n, p) for i in range(k + 1))
    
    @staticmethod
    def binomial_sample(n: int, p: float, size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Binomial distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.binomial(n, p, size)
    
    @staticmethod
    def binomial_stats(n: int, p: float) -> DistributionStats:
        """Compute statistics of Binomial distribution."""
        mean = n * p
        variance = n * p * (1 - p)
        std = np.sqrt(variance)
        skewness = (1 - 2 * p) / np.sqrt(n * p * (1 - p))
        kurtosis = (1 - 6 * p * (1 - p)) / (n * p * (1 - p))
        
        return DistributionStats(mean, variance, std, skewness, kurtosis)
    
    @staticmethod
    def poisson_pmf(k: int, lambda_: float) -> float:
        """
        Poisson distribution probability mass function.
        
        Args:
            k: Number of events
            lambda_: Average rate
            
        Returns:
            P(X = k)
        """
        if k < 0:
            return 0.0
        return (lambda_ ** k) * np.exp(-lambda_) / Probability.factorial(k)
    
    @staticmethod
    def poisson_cdf(k: int, lambda_: float) -> float:
        """Poisson cumulative distribution function."""
        return sum(Probability.poisson_pmf(i, lambda_) for i in range(k + 1))
    
    @staticmethod
    def poisson_sample(lambda_: float, size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Poisson distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.poisson(lambda_, size)
    
    @staticmethod
    def poisson_stats(lambda_: float) -> DistributionStats:
        """Compute statistics of Poisson distribution."""
        mean = lambda_
        variance = lambda_
        std = np.sqrt(lambda_)
        skewness = 1 / np.sqrt(lambda_)
        kurtosis = 1 / lambda_
        
        return DistributionStats(mean, variance, std, skewness, kurtosis)
    
    @staticmethod
    def geometric_pmf(k: int, p: float) -> float:
        """
        Geometric distribution probability mass function.
        
        Args:
            k: Number of trials until first success
            p: Probability of success
            
        Returns:
            P(X = k)
        """
        if k < 1:
            return 0.0
        return (1 - p) ** (k - 1) * p
    
    @staticmethod
    def geometric_sample(p: float, size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Geometric distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.geometric(p, size)
    
    @staticmethod
    def negative_binomial_pmf(k: int, r: int, p: float) -> float:
        """
        Negative Binomial distribution PMF.
        
        Args:
            k: Number of failures before r successes
            r: Number of successes
            p: Probability of success
            
        Returns:
            P(X = k)
        """
        if k < 0:
            return 0.0
        coeff = Probability.binomial_coefficient(k + r - 1, k)
        return coeff * (p ** r) * ((1 - p) ** k)
    
    # ==================== Continuous Distributions ====================
    
    @staticmethod
    def uniform_pdf(x: float, a: float, b: float) -> float:
        """
        Uniform distribution probability density function.
        
        Args:
            x: Point
            a: Lower bound
            b: Upper bound
            
        Returns:
            f(x)
        """
        if a <= x <= b:
            return 1.0 / (b - a)
        return 0.0
    
    @staticmethod
    def uniform_cdf(x: float, a: float, b: float) -> float:
        """Uniform cumulative distribution function."""
        if x < a:
            return 0.0
        elif x > b:
            return 1.0
        return (x - a) / (b - a)
    
    @staticmethod
    def uniform_sample(a: float, b: float, size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Uniform distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(a, b, size)
    
    @staticmethod
    def uniform_stats(a: float, b: float) -> DistributionStats:
        """Compute statistics of Uniform distribution."""
        mean = (a + b) / 2
        variance = ((b - a) ** 2) / 12
        std = np.sqrt(variance)
        skewness = 0.0
        kurtosis = -1.2
        
        return DistributionStats(mean, variance, std, skewness, kurtosis)
    
    @staticmethod
    def gaussian_pdf(x: Union[float, np.ndarray], mu: float = 0.0, 
                    sigma: float = 1.0) -> Union[float, np.ndarray]:
        """
        Gaussian (Normal) distribution probability density function.
        
        Args:
            x: Point(s)
            mu: Mean
            sigma: Standard deviation
            
        Returns:
            f(x)
        """
        coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)
    
    @staticmethod
    def gaussian_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Gaussian cumulative distribution function.
        
        Uses error function approximation.
        """
        z = (x - mu) / sigma
        return 0.5 * (1 + np.erf(z / np.sqrt(2)))
    
    @staticmethod
    def gaussian_sample(mu: float = 0.0, sigma: float = 1.0, 
                       size: int = 1, seed: int = None) -> np.ndarray:
        """Sample from Gaussian distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mu, sigma, size)
    
    @staticmethod
    def gaussian_stats(mu: float, sigma: float) -> DistributionStats:
        """Compute statistics of Gaussian distribution."""
        return DistributionStats(
            mean=mu,
            variance=sigma ** 2,
            std=sigma,
            skewness=0.0,
            kurtosis=0.0
        )
    
    @staticmethod
    def multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, 
                                  Sigma: np.ndarray) -> float:
        """
        Multivariate Gaussian probability density function.
        
        Args:
            x: Point (n,)
            mu: Mean vector (n,)
            Sigma: Covariance matrix (n x n)
            
        Returns:
            f(x)
        """
        n = len(mu)
        diff = x - mu
        
        # Compute determinant and inverse
        det = np.linalg.det(Sigma)
        if det <= 0:
            raise ValueError("Covariance matrix must be positive definite")
        
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Compute PDF
        coeff = 1.0 / (np.sqrt((2 * np.pi) ** n * det))
        exponent = -0.5 * diff.T @ Sigma_inv @ diff
        
        return coeff * np.exp(exponent)
    
    @staticmethod
    def exponential_pdf(x: float, lambda_: float) -> float:
        """
        Exponential distribution probability density function.
        
        Args:
            x: Point (x ≥ 0)
            lambda_: Rate parameter
            
        Returns:
            f(x)
        """
        if x < 0:
            return 0.0
        return lambda_ * np.exp(-lambda_ * x)
    
    @staticmethod
    def exponential_cdf(x: float, lambda_: float) -> float:
        """Exponential cumulative distribution function."""
        if x < 0:
            return 0.0
        return 1 - np.exp(-lambda_ * x)
    
    @staticmethod
    def exponential_sample(lambda_: float, size: int = 1, 
                          seed: int = None) -> np.ndarray:
        """Sample from Exponential distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.exponential(1 / lambda_, size)
    
    @staticmethod
    def exponential_stats(lambda_: float) -> DistributionStats:
        """Compute statistics of Exponential distribution."""
        mean = 1 / lambda_
        variance = 1 / (lambda_ ** 2)
        std = 1 / lambda_
        skewness = 2.0
        kurtosis = 6.0
        
        return DistributionStats(mean, variance, std, skewness, kurtosis)
    
    @staticmethod
    def gamma_pdf(x: float, alpha: float, beta: float) -> float:
        """
        Gamma distribution probability density function.
        
        Args:
            x: Point (x ≥ 0)
            alpha: Shape parameter
            beta: Rate parameter
            
        Returns:
            f(x)
        """
        if x < 0:
            return 0.0
        
        # Gamma function approximation using scipy
        from scipy.special import gamma as gamma_func
        
        coeff = (beta ** alpha) / gamma_func(alpha)
        return coeff * (x ** (alpha - 1)) * np.exp(-beta * x)
    
    @staticmethod
    def gamma_sample(alpha: float, beta: float, size: int = 1,
                    seed: int = None) -> np.ndarray:
        """Sample from Gamma distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.gamma(alpha, 1 / beta, size)
    
    @staticmethod
    def beta_pdf(x: float, alpha: float, beta: float) -> float:
        """
        Beta distribution probability density function.
        
        Args:
            x: Point (0 ≤ x ≤ 1)
            alpha: Shape parameter α
            beta: Shape parameter β
            
        Returns:
            f(x)
        """
        if x < 0 or x > 1:
            return 0.0
        
        from scipy.special import beta as beta_func
        
        coeff = 1 / beta_func(alpha, beta)
        return coeff * (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))
    
    @staticmethod
    def beta_sample(alpha: float, beta: float, size: int = 1,
                   seed: int = None) -> np.ndarray:
        """Sample from Beta distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.beta(alpha, beta, size)
    
    @staticmethod
    def chi_squared_pdf(x: float, k: int) -> float:
        """
        Chi-squared distribution probability density function.
        
        Args:
            x: Point (x ≥ 0)
            k: Degrees of freedom
            
        Returns:
            f(x)
        """
        if x < 0:
            return 0.0
        
        from scipy.special import gamma as gamma_func
        
        coeff = 1 / (2 ** (k / 2) * gamma_func(k / 2))
        return coeff * (x ** (k / 2 - 1)) * np.exp(-x / 2)
    
    @staticmethod
    def chi_squared_sample(k: int, size: int = 1,
                          seed: int = None) -> np.ndarray:
        """Sample from Chi-squared distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.chisquare(k, size)
    
    @staticmethod
    def student_t_pdf(x: float, nu: float) -> float:
        """
        Student's t-distribution probability density function.
        
        Args:
            x: Point
            nu: Degrees of freedom
            
        Returns:
            f(x)
        """
        from scipy.special import gamma as gamma_func
        
        coeff = gamma_func((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma_func(nu / 2))
        return coeff * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)
    
    # ==================== Bayes' Theorem ====================
    
    @staticmethod
    def bayes_theorem(p_a: float, p_b_given_a: float, p_b: float) -> float:
        """
        Apply Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
        
        Args:
            p_a: Prior probability P(A)
            p_b_given_a: Likelihood P(B|A)
            p_b: Marginal probability P(B)
            
        Returns:
            Posterior probability P(A|B)
        """
        if p_b == 0:
            raise ValueError("P(B) must be non-zero")
        return (p_b_given_a * p_a) / p_b
    
    @staticmethod
    def bayes_theorem_full(p_a: float, p_b_given_a: float, 
                          p_b_given_not_a: float) -> float:
        """
        Apply Bayes' theorem with full computation of P(B).
        
        Args:
            p_a: Prior P(A)
            p_b_given_a: Likelihood P(B|A)
            p_b_given_not_a: Likelihood P(B|¬A)
            
        Returns:
            Posterior P(A|B)
        """
        p_not_a = 1 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
        return Probability.bayes_theorem(p_a, p_b_given_a, p_b)
    
    @staticmethod
    def naive_bayes_classifier(priors: dict, likelihoods: dict, 
                              evidence: dict) -> dict:
        """
        Simple Naive Bayes classifier.
        
        Args:
            priors: Dictionary of P(class) for each class
            likelihoods: Dictionary of P(feature|class) for each class and feature
            evidence: Dictionary of observed feature values
            
        Returns:
            Dictionary of posterior probabilities for each class
        """
        posteriors = {}
        
        for class_label, prior in priors.items():
            posterior = prior
            for feature, value in evidence.items():
                likelihood = likelihoods.get((class_label, feature, value), 0.01)
                posterior *= likelihood
            posteriors[class_label] = posterior
        
        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        
        return posteriors
    
    # ==================== Conditional Probability ====================
    
    @staticmethod
    def conditional_probability(p_a_and_b: float, p_b: float) -> float:
        """
        Compute conditional probability P(A|B) = P(A ∩ B) / P(B).
        
        Args:
            p_a_and_b: Joint probability P(A ∩ B)
            p_b: Marginal probability P(B)
            
        Returns:
            Conditional probability P(A|B)
        """
        if p_b == 0:
            raise ValueError("P(B) must be non-zero")
        return p_a_and_b / p_b
    
    @staticmethod
    def joint_probability_independent(p_a: float, p_b: float) -> float:
        """Compute joint probability for independent events."""
        return p_a * p_b
    
    @staticmethod
    def law_of_total_probability(p_a_given_b: List[float], 
                                 p_b: List[float]) -> float:
        """
        Compute P(A) using law of total probability.
        
        Args:
            p_a_given_b: List of P(A|B_i) for each partition
            p_b: List of P(B_i) for each partition
            
        Returns:
            P(A) = Σ P(A|B_i) * P(B_i)
        """
        if len(p_a_given_b) != len(p_b):
            raise ValueError("Lists must have same length")
        return sum(pa_b * pb for pa_b, pb in zip(p_a_given_b, p_b))
    
    # ==================== Expectation and Variance ====================
    
    @staticmethod
    def expectation(values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Compute expected value E[X] = Σ x * P(X=x).
        
        Args:
            values: Possible values of X
            probabilities: Corresponding probabilities
            
        Returns:
            Expected value
        """
        if len(values) != len(probabilities):
            raise ValueError("Arrays must have same length")
        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1")
        
        return np.sum(values * probabilities)
    
    @staticmethod
    def expectation_continuous(f: Callable[[float], float], 
                              pdf: Callable[[float], float],
                              x_range: Tuple[float, float],
                              n_points: int = 10000) -> float:
        """
        Compute expected value for continuous distribution.
        
        E[X] = ∫ x * f(x) dx
        
        Args:
            f: Function to compute expectation of (usually f(x) = x)
            pdf: Probability density function
            x_range: Integration range
            n_points: Number of points for numerical integration
            
        Returns:
            Expected value
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        dx = (x_range[1] - x_range[0]) / n_points
        
        integrand = np.array([f(xi) * pdf(xi) for xi in x])
        return np.sum(integrand) * dx
    
    @staticmethod
    def variance(values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Compute variance Var(X) = E[(X - μ)²].
        
        Args:
            values: Possible values of X
            probabilities: Corresponding probabilities
            
        Returns:
            Variance
        """
        mu = Probability.expectation(values, probabilities)
        return Probability.expectation((values - mu) ** 2, probabilities)
    
    @staticmethod
    def covariance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute covariance Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)].
        
        Args:
            x: Samples of X
            y: Samples of Y
            
        Returns:
            Covariance
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have same length")
        
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        
        return np.mean((x - mu_x) * (y - mu_y))
    
    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            x: Samples of X
            y: Samples of Y
            
        Returns:
            Correlation in range [-1, 1]
        """
        cov = Probability.covariance(x, y)
        std_x = np.std(x)
        std_y = np.std(y)
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return cov / (std_x * std_y)
    
    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix for multivariate data.
        
        Args:
            data: Data matrix (n_samples x n_features)
            
        Returns:
            Correlation matrix (n_features x n_features)
        """
        return np.corrcoef(data.T)
    
    # ==================== Maximum Likelihood Estimation ====================
    
    @staticmethod
    def mle_gaussian(data: np.ndarray) -> Tuple[float, float]:
        """
        Maximum Likelihood Estimation for Gaussian parameters.
        
        Args:
            data: Observed data
            
        Returns:
            Tuple of (mu_MLE, sigma_MLE)
        """
        mu_mle = np.mean(data)
        sigma_mle = np.sqrt(np.mean((data - mu_mle) ** 2))
        return mu_mle, sigma_mle
    
    @staticmethod
    def mle_bernoulli(data: np.ndarray) -> float:
        """
        MLE for Bernoulli parameter.
        
        Args:
            data: Binary observations (0s and 1s)
            
        Returns:
            p_MLE
        """
        return np.mean(data)
    
    @staticmethod
    def mle_poisson(data: np.ndarray) -> float:
        """
        MLE for Poisson parameter.
        
        Args:
            data: Count data
            
        Returns:
            lambda_MLE
        """
        return np.mean(data)
    
    @staticmethod
    def mle_exponential(data: np.ndarray) -> float:
        """
        MLE for Exponential parameter.
        
        Args:
            data: Positive observations
            
        Returns:
            lambda_MLE
        """
        return 1 / np.mean(data)
    
    @staticmethod
    def log_likelihood_gaussian(data: np.ndarray, mu: float, 
                               sigma: float) -> float:
        """
        Compute log-likelihood for Gaussian distribution.
        
        Args:
            data: Observed data
            mu: Mean parameter
            sigma: Standard deviation parameter
            
        Returns:
            Log-likelihood
        """
        n = len(data)
        ll = -n / 2 * np.log(2 * np.pi * sigma ** 2)
        ll -= np.sum((data - mu) ** 2) / (2 * sigma ** 2)
        return ll
    
    # ==================== Maximum A Posteriori ====================
    
    @staticmethod
    def map_gaussian(data: np.ndarray, prior_mu: float, 
                    prior_sigma: float, known_sigma: float) -> float:
        """
        MAP estimation for Gaussian mean with Gaussian prior.
        
        Args:
            data: Observed data
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
            known_sigma: Known data standard deviation
            
        Returns:
            mu_MAP
        """
        n = len(data)
        data_mean = np.mean(data)
        
        # Posterior precision = prior precision + data precision
        prior_precision = 1 / prior_sigma ** 2
        data_precision = n / known_sigma ** 2
        
        # MAP is weighted average
        weight = data_precision / (prior_precision + data_precision)
        
        return (1 - weight) * prior_mu + weight * data_mean
    
    # ==================== Sampling Methods ====================
    
    @staticmethod
    def inverse_transform_sampling(cdf: Callable[[float], float], 
                                   n_samples: int = 1,
                                   seed: int = None) -> np.ndarray:
        """
        Generate samples using inverse transform sampling.
        
        Args:
            cdf: Cumulative distribution function
            n_samples: Number of samples
            seed: Random seed
            
        Returns:
            Samples from the distribution
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate uniform samples
        u = np.random.uniform(0, 1, n_samples)
        
        # Binary search for inverse CDF
        samples = []
        for ui in u:
            # Simple binary search implementation
            low, high = -100, 100
            for _ in range(50):
                mid = (low + high) / 2
                if cdf(mid) < ui:
                    low = mid
                else:
                    high = mid
            samples.append((low + high) / 2)
        
        return np.array(samples)
    
    @staticmethod
    def rejection_sampling(target_pdf: Callable[[float], float],
                          proposal_pdf: Callable[[float], float],
                          proposal_sample: Callable[[int], np.ndarray],
                          M: float, n_samples: int = 1,
                          seed: int = None) -> np.ndarray:
        """
        Generate samples using rejection sampling.
        
        Args:
            target_pdf: Target distribution PDF
            proposal_pdf: Proposal distribution PDF
            proposal_sample: Function to sample from proposal
            M: Constant such that target_pdf(x) ≤ M * proposal_pdf(x)
            n_samples: Number of samples to generate
            seed: Random seed
            
        Returns:
            Samples from target distribution
        """
        if seed is not None:
            np.random.seed(seed)
        
        samples = []
        while len(samples) < n_samples:
            # Sample from proposal
            x = proposal_sample(1)[0]
            u = np.random.uniform(0, M * proposal_pdf(x))
            
            # Accept/reject
            if u <= target_pdf(x):
                samples.append(x)
        
        return np.array(samples)
    
    @staticmethod
    def metropolis_hastings(target_log_pdf: Callable[[float], float],
                           proposal_std: float, n_samples: int = 1000,
                           initial: float = 0.0, seed: int = None) -> np.ndarray:
        """
        Generate samples using Metropolis-Hastings MCMC.
        
        Args:
            target_log_pdf: Log of target PDF
            proposal_std: Standard deviation of Gaussian proposal
            n_samples: Number of samples
            initial: Initial value
            seed: Random seed
            
        Returns:
            Samples from target distribution
        """
        if seed is not None:
            np.random.seed(seed)
        
        samples = np.zeros(n_samples)
        current = initial
        current_log_prob = target_log_pdf(current)
        
        for i in range(n_samples):
            # Propose new state
            proposal = current + np.random.normal(0, proposal_std)
            proposal_log_prob = target_log_pdf(proposal)
            
            # Acceptance probability
            log_alpha = proposal_log_prob - current_log_prob
            
            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                current = proposal
                current_log_prob = proposal_log_prob
            
            samples[i] = current
        
        return samples
    
    @staticmethod
    def gibbs_sampling(conditionals: List[Callable], n_samples: int = 1000,
                      initial: np.ndarray = None, burn_in: int = 100,
                      seed: int = None) -> np.ndarray:
        """
        Generate samples using Gibbs sampling.
        
        Args:
            conditionals: List of conditional sampling functions
            n_samples: Number of samples
            initial: Initial state
            burn_in: Burn-in period
            seed: Random seed
            
        Returns:
            Samples after burn-in
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_dims = len(conditionals)
        if initial is None:
            initial = np.zeros(n_dims)
        
        samples = []
        current = initial.copy()
        
        for _ in range(n_samples + burn_in):
            # Sample each dimension conditionally
            for i in range(n_dims):
                current[i] = conditionals[i](current, i)
            
            if _ >= burn_in:
                samples.append(current.copy())
        
        return np.array(samples)
    
    # ==================== Information Theory ====================
    
    @staticmethod
    def entropy(probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy H(X) = -Σ p(x) log p(x).
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Entropy in nats (use log2 for bits)
        """
        # Filter out zero probabilities
        p = probabilities[probabilities > 0]
        return -np.sum(p * np.log(p))
    
    @staticmethod
    def entropy_bits(probabilities: np.ndarray) -> float:
        """Compute entropy in bits (using log base 2)."""
        p = probabilities[probabilities > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute cross entropy H(p, q) = -Σ p(x) log q(x).
        
        Args:
            p: True distribution
            q: Approximating distribution
            
        Returns:
            Cross entropy
        """
        # Avoid log(0)
        q = np.clip(q, 1e-15, 1)
        return -np.sum(p * np.log(q))
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence D_KL(p||q) = Σ p(x) log(p(x)/q(x)).
        
        Args:
            p: True distribution
            q: Approximating distribution
            
        Returns:
            KL divergence (non-negative)
        """
        # Avoid division by zero and log(0)
        p = np.clip(p, 1e-15, 1)
        q = np.clip(q, 1e-15, 1)
        return np.sum(p * np.log(p / q))
    
    @staticmethod
    def mutual_information(joint: np.ndarray, marginal_x: np.ndarray,
                          marginal_y: np.ndarray) -> float:
        """
        Compute mutual information I(X;Y).
        
        Args:
            joint: Joint probability matrix P(X,Y)
            marginal_x: Marginal P(X)
            marginal_y: Marginal P(Y)
            
        Returns:
            Mutual information
        """
        # Outer product for independence
        independent = np.outer(marginal_x, marginal_y)
        
        # Avoid log(0)
        joint = np.clip(joint, 1e-15, 1)
        independent = np.clip(independent, 1e-15, 1)
        
        return np.sum(joint * np.log(joint / independent))
    
    # ==================== Hypothesis Testing ====================
    
    @staticmethod
    def z_test(sample: np.ndarray, pop_mean: float, 
              pop_std: float) -> Tuple[float, float]:
        """
        One-sample Z-test.
        
        Args:
            sample: Sample data
            pop_mean: Population mean under null hypothesis
            pop_std: Population standard deviation
            
        Returns:
            Tuple of (z-statistic, two-tailed p-value)
        """
        n = len(sample)
        sample_mean = np.mean(sample)
        
        z = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
        
        # Two-tailed p-value
        p_value = 2 * (1 - Probability.gaussian_cdf(abs(z)))
        
        return z, p_value
    
    @staticmethod
    def t_test_one_sample(sample: np.ndarray, 
                         pop_mean: float) -> Tuple[float, float]:
        """
        One-sample t-test.
        
        Args:
            sample: Sample data
            pop_mean: Population mean under null hypothesis
            
        Returns:
            Tuple of (t-statistic, two-tailed p-value)
        """
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        t_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
        
        # Use scipy for t-distribution CDF
        from scipy.stats import t
        p_value = 2 * (1 - t.cdf(abs(t_stat), n - 1))
        
        return t_stat, p_value
    
    @staticmethod
    def t_test_two_sample(sample1: np.ndarray, 
                         sample2: np.ndarray,
                         equal_var: bool = True) -> Tuple[float, float]:
        """
        Two-sample t-test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            equal_var: Assume equal variances
            
        Returns:
            Tuple of (t-statistic, two-tailed p-value)
        """
        from scipy.stats import ttest_ind
        
        t_stat, p_value = ttest_ind(sample1, sample2, equal_var=equal_var)
        return t_stat, p_value
    
    @staticmethod
    def chi_squared_test(observed: np.ndarray, 
                        expected: np.ndarray) -> Tuple[float, float]:
        """
        Chi-squared goodness of fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            
        Returns:
            Tuple of (chi-squared statistic, p-value)
        """
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1
        
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi2, df)
        
        return chi2, p_value
