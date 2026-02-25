"""
Statistics Module

Statistical methods and inference for machine learning and data analysis.
Includes descriptive statistics, hypothesis testing, and regression analysis.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Dict
from scipy import stats as scipy_stats
from dataclasses import dataclass


@dataclass
class SummaryStatistics:
    """Summary statistics for a dataset."""
    count: int
    mean: float
    std: float
    std_error: float
    median: float
    min: float
    max: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    skewness: float
    kurtosis: float


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    lower: float
    upper: float
    confidence_level: float
    estimate: float


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    statistic: float
    p_value: float
    reject_null: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[int] = None


class Statistics:
    """
    Comprehensive statistical methods for data analysis.
    
    Features:
    - Descriptive statistics
    - Confidence intervals
    - Hypothesis testing
    - ANOVA
    - Non-parametric tests
    - Correlation and association
    - Regression diagnostics
    - Resampling methods
    """
    
    # ==================== Descriptive Statistics ====================
    
    @staticmethod
    def summary(data: np.ndarray) -> SummaryStatistics:
        """
        Compute comprehensive summary statistics.
        
        Args:
            data: Input data array
            
        Returns:
            SummaryStatistics object
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]  # Remove NaN values
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1) if n > 1 else 0.0
        std_error = std / np.sqrt(n) if n > 0 else 0.0
        median = np.median(data)
        min_val = np.min(data)
        max_val = np.max(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        skewness = scipy_stats.skew(data) if n > 2 else 0.0
        kurtosis = scipy_stats.kurtosis(data) if n > 3 else 0.0
        
        return SummaryStatistics(
            count=n, mean=mean, std=std, std_error=std_error,
            median=median, min=min_val, max=max_val,
            q1=q1, q3=q3, iqr=iqr,
            skewness=skewness, kurtosis=kurtosis
        )
    
    @staticmethod
    def mean(data: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
        """Compute arithmetic mean."""
        return np.mean(data, axis=axis)
    
    @staticmethod
    def median(data: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
        """Compute median."""
        return np.median(data, axis=axis)
    
    @staticmethod
    def mode(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mode (most frequent values).
        
        Returns:
            Tuple of (mode values, their counts)
        """
        values, counts = np.unique(data, return_counts=True)
        max_count = np.max(counts)
        mode_mask = counts == max_count
        return values[mode_mask], counts[mode_mask]
    
    @staticmethod
    def variance(data: np.ndarray, ddof: int = 1, axis: int = None) -> Union[float, np.ndarray]:
        """
        Compute variance.
        
        Args:
            data: Input data
            ddof: Delta degrees of freedom (use 0 for population, 1 for sample)
            axis: Axis along which to compute
            
        Returns:
            Variance
        """
        return np.var(data, axis=axis, ddof=ddof)
    
    @staticmethod
    def std(data: np.ndarray, ddof: int = 1, axis: int = None) -> Union[float, np.ndarray]:
        """Compute standard deviation."""
        return np.std(data, axis=axis, ddof=ddof)
    
    @staticmethod
    def coefficient_of_variation(data: np.ndarray) -> float:
        """Compute coefficient of variation (CV = std/mean)."""
        mean = np.mean(data)
        if mean == 0:
            return np.inf
        return np.std(data, ddof=1) / mean
    
    @staticmethod
    def percentiles(data: np.ndarray, percentiles: List[float] = [25, 50, 75]) -> np.ndarray:
        """Compute specified percentiles."""
        return np.percentile(data, percentiles)
    
    @staticmethod
    def iqr(data: np.ndarray) -> float:
        """Compute interquartile range."""
        return np.percentile(data, 75) - np.percentile(data, 25)
    
    @staticmethod
    def skewness(data: np.ndarray) -> float:
        """Compute sample skewness."""
        n = len(data)
        if n <= 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2)) * (n - 1)
    
    @staticmethod
    def kurtosis(data: np.ndarray, excess: bool = True) -> float:
        """
        Compute kurtosis.
        
        Args:
            data: Input data
            excess: If True, return excess kurtosis (kurtosis - 3)
            
        Returns:
            Kurtosis
        """
        n = len(data)
        if n <= 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        
        m4 = np.mean((data - mean) ** 4)
        m2 = np.mean((data - mean) ** 2)
        
        if m2 == 0:
            return 0.0
        
        kurt = m4 / (m2 ** 2)
        return kurt - 3 if excess else kurt
    
    @staticmethod
    def z_scores(data: np.ndarray) -> np.ndarray:
        """
        Compute z-scores (standardize data).
        
        Args:
            data: Input data
            
        Returns:
            Z-scores
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    @staticmethod
    def robust_z_scores(data: np.ndarray, method: str = "mad") -> np.ndarray:
        """
        Compute robust z-scores using median and MAD.
        
        Args:
            data: Input data
            method: 'mad' (Median Absolute Deviation) or 'iqr'
            
        Returns:
            Robust z-scores
        """
        median = np.median(data)
        
        if method == "mad":
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return np.zeros_like(data)
            return 0.6745 * (data - median) / mad
        elif method == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return np.zeros_like(data)
            return (data - median) / iqr
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ==================== Confidence Intervals ====================
    
    @staticmethod
    def confidence_interval_mean(data: np.ndarray, 
                                 confidence_level: float = 0.95) -> ConfidenceInterval:
        """
        Compute confidence interval for the mean.
        
        Args:
            data: Sample data
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            ConfidenceInterval object
        """
        n = len(data)
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(n)
        
        # Use t-distribution for small samples
        df = n - 1
        alpha = 1 - confidence_level
        t_critical = scipy_stats.t.ppf(1 - alpha / 2, df)
        
        margin = t_critical * std_error
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
            estimate=mean
        )
    
    @staticmethod
    def confidence_interval_proportion(successes: int, n: int,
                                      confidence_level: float = 0.95) -> ConfidenceInterval:
        """
        Compute confidence interval for a proportion.
        
        Args:
            successes: Number of successes
            n: Sample size
            confidence_level: Confidence level
            
        Returns:
            ConfidenceInterval object
        """
        p_hat = successes / n
        
        # Wilson score interval
        z = scipy_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        denominator = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denominator
        
        return ConfidenceInterval(
            lower=max(0, center - margin),
            upper=min(1, center + margin),
            confidence_level=confidence_level,
            estimate=p_hat
        )
    
    @staticmethod
    def confidence_interval_difference_means(data1: np.ndarray, data2: np.ndarray,
                                             confidence_level: float = 0.95,
                                             equal_var: bool = False) -> ConfidenceInterval:
        """
        Compute confidence interval for difference of means.
        
        Args:
            data1: First sample
            data2: Second sample
            confidence_level: Confidence level
            equal_var: Assume equal variances
            
        Returns:
            ConfidenceInterval object
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        diff = mean1 - mean2
        
        if equal_var:
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's approximation
            se = np.sqrt(var1/n1 + var2/n2)
            # Welch-Satterthwaite degrees of freedom
            num = (var1/n1 + var2/n2) ** 2
            denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df = num / denom
        
        alpha = 1 - confidence_level
        t_critical = scipy_stats.t.ppf(1 - alpha / 2, df)
        margin = t_critical * se
        
        return ConfidenceInterval(
            lower=diff - margin,
            upper=diff + margin,
            confidence_level=confidence_level,
            estimate=diff
        )
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, statistic: callable,
                                      n_bootstrap: int = 1000,
                                      confidence_level: float = 0.95,
                                      seed: int = None) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Sample data
            statistic: Function to compute statistic of interest
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            seed: Random seed
            
        Returns:
            ConfidenceInterval object
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(data)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            bootstrap_stats[i] = statistic(bootstrap_sample)
        
        # Percentile method
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            estimate=statistic(data)
        )
    
    # ==================== Hypothesis Testing ====================
    
    @staticmethod
    def z_test_one_sample(sample: np.ndarray, pop_mean: float,
                          pop_std: float, alternative: str = "two-sided") -> HypothesisTestResult:
        """
        One-sample Z-test (known population standard deviation).
        
        Args:
            sample: Sample data
            pop_mean: Hypothesized population mean
            pop_std: Known population standard deviation
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        n = len(sample)
        sample_mean = np.mean(sample)
        
        z = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
        
        if alternative == "two-sided":
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        elif alternative == "greater":
            p_value = 1 - scipy_stats.norm.cdf(z)
        elif alternative == "less":
            p_value = scipy_stats.norm.cdf(z)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        return HypothesisTestResult(
            statistic=z,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def t_test_one_sample(sample: np.ndarray, pop_mean: float,
                          alternative: str = "two-sided") -> HypothesisTestResult:
        """
        One-sample t-test.
        
        Args:
            sample: Sample data
            pop_mean: Hypothesized population mean
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        t_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
        df = n - 1
        
        if alternative == "two-sided":
            p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
        elif alternative == "greater":
            p_value = 1 - scipy_stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = scipy_stats.t.cdf(t_stat, df)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # Confidence interval
        alpha = 0.05
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
        margin = t_crit * sample_std / np.sqrt(n)
        ci = (sample_mean - margin, sample_mean + margin)
        
        return HypothesisTestResult(
            statistic=t_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            confidence_interval=ci,
            degrees_of_freedom=df
        )
    
    @staticmethod
    def t_test_two_sample(sample1: np.ndarray, sample2: np.ndarray,
                          equal_var: bool = True, alternative: str = "two-sided") -> HypothesisTestResult:
        """
        Two-sample t-test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            equal_var: Assume equal variances
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import ttest_ind
        
        t_stat, p_value = ttest_ind(sample1, sample2, equal_var=equal_var, alternative=alternative)
        
        # Degrees of freedom
        n1, n2 = len(sample1), len(sample2)
        if equal_var:
            df = n1 + n2 - 2
        else:
            # Welch-Satterthwaite
            var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
            num = (var1/n1 + var2/n2) ** 2
            denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df = int(num / denom)
        
        return HypothesisTestResult(
            statistic=t_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=df
        )
    
    @staticmethod
    def t_test_paired(sample1: np.ndarray, sample2: np.ndarray,
                      alternative: str = "two-sided") -> HypothesisTestResult:
        """
        Paired t-test.
        
        Args:
            sample1: First sample (paired)
            sample2: Second sample (paired)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import ttest_rel
        
        t_stat, p_value = ttest_rel(sample1, sample2, alternative=alternative)
        df = len(sample1) - 1
        
        return HypothesisTestResult(
            statistic=t_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=df
        )
    
    @staticmethod
    def anova_one_way(*groups: np.ndarray) -> HypothesisTestResult:
        """
        One-way ANOVA.
        
        Args:
            *groups: Variable number of group arrays
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import f_oneway
        
        f_stat, p_value = f_oneway(*groups)
        
        k = len(groups)  # Number of groups
        n_total = sum(len(g) for g in groups)
        df_between = k - 1
        df_within = n_total - k
        
        return HypothesisTestResult(
            statistic=f_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=df_between
        )
    
    @staticmethod
    def anova_two_way(data: np.ndarray, factor1: np.ndarray, 
                      factor2: np.ndarray) -> Dict:
        """
        Two-way ANOVA.
        
        Args:
            data: Response variable
            factor1: First factor labels
            factor2: Second factor labels
            
        Returns:
            Dictionary with results for both factors and interaction
        """
        import pandas as pd
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        
        df = pd.DataFrame({
            'value': data,
            'factor1': factor1,
            'factor2': factor2
        })
        
        model = ols('value ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', data=df).fit()
        anova_table = anova_lm(model)
        
        return {
            'factor1': HypothesisTestResult(
                statistic=anova_table['F'][0],
                p_value=anova_table['PR(>F)'][0],
                reject_null=anova_table['PR(>F)'][0] < 0.05
            ),
            'factor2': HypothesisTestResult(
                statistic=anova_table['F'][1],
                p_value=anova_table['PR(>F)'][1],
                reject_null=anova_table['PR(>F)'][1] < 0.05
            ),
            'interaction': HypothesisTestResult(
                statistic=anova_table['F'][2],
                p_value=anova_table['PR(>F)'][2],
                reject_null=anova_table['PR(>F)'][2] < 0.05
            )
        }
    
    @staticmethod
    def chi_squared_test(observed: np.ndarray, 
                         expected: Optional[np.ndarray] = None) -> HypothesisTestResult:
        """
        Chi-squared goodness of fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (if None, assumes uniform)
            
        Returns:
            HypothesisTestResult
        """
        if expected is None:
            expected = np.full(len(observed), np.sum(observed) / len(observed))
        
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1
        
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df)
        
        return HypothesisTestResult(
            statistic=chi2,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=df
        )
    
    @staticmethod
    def chi_squared_test_independence(contingency_table: np.ndarray) -> HypothesisTestResult:
        """
        Chi-squared test of independence.
        
        Args:
            contingency_table: Contingency table (2D array)
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import chi2_contingency
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return HypothesisTestResult(
            statistic=chi2,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=dof
        )
    
    # ==================== Non-parametric Tests ====================
    
    @staticmethod
    def mann_whitney_u(sample1: np.ndarray, sample2: np.ndarray,
                       alternative: str = "two-sided") -> HypothesisTestResult:
        """
        Mann-Whitney U test (Wilcoxon rank-sum test).
        
        Args:
            sample1: First sample
            sample2: Second sample
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import mannwhitneyu
        
        u_stat, p_value = mannwhitneyu(sample1, sample2, alternative=alternative)
        
        return HypothesisTestResult(
            statistic=u_stat,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def wilcoxon_signed_rank(sample1: np.ndarray, sample2: np.ndarray,
                             alternative: str = "two-sided") -> HypothesisTestResult:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Args:
            sample1: First sample (paired)
            sample2: Second sample (paired)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import wilcoxon
        
        w_stat, p_value = wilcoxon(sample1, sample2, alternative=alternative)
        
        return HypothesisTestResult(
            statistic=w_stat,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def kruskal_wallis(*groups: np.ndarray) -> HypothesisTestResult:
        """
        Kruskal-Wallis H test (non-parametric one-way ANOVA).
        
        Args:
            *groups: Variable number of group arrays
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import kruskal
        
        h_stat, p_value = kruskal(*groups)
        
        return HypothesisTestResult(
            statistic=h_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=len(groups) - 1
        )
    
    @staticmethod
    def friedman_test(*groups: np.ndarray) -> HypothesisTestResult:
        """
        Friedman test (non-parametric repeated measures ANOVA).
        
        Args:
            *groups: Related samples (blocks)
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import friedmanchisquare
        
        chi2_stat, p_value = friedmanchisquare(*groups)
        
        return HypothesisTestResult(
            statistic=chi2_stat,
            p_value=p_value,
            reject_null=p_value < 0.05,
            degrees_of_freedom=len(groups) - 1
        )
    
    @staticmethod
    def kolmogorov_smirnov_test(sample1: np.ndarray, 
                                sample2: Optional[np.ndarray] = None) -> HypothesisTestResult:
        """
        Kolmogorov-Smirnov test.
        
        Args:
            sample1: First sample (or reference distribution if sample2 provided)
            sample2: Second sample (optional, for two-sample test)
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import kstest, ks_2samp
        
        if sample2 is None:
            # One-sample KS test against normal distribution
            ks_stat, p_value = kstest(sample1, 'norm')
        else:
            # Two-sample KS test
            ks_stat, p_value = ks_2samp(sample1, sample2)
        
        return HypothesisTestResult(
            statistic=ks_stat,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def shapiro_wilk_test(sample: np.ndarray) -> HypothesisTestResult:
        """
        Shapiro-Wilk test for normality.
        
        Args:
            sample: Sample data
            
        Returns:
            HypothesisTestResult
        """
        from scipy.stats import shapiro
        
        w_stat, p_value = shapiro(sample)
        
        return HypothesisTestResult(
            statistic=w_stat,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def anderson_darling_test(sample: np.ndarray, 
                              dist: str = 'norm') -> Dict:
        """
        Anderson-Darling test for normality.
        
        Args:
            sample: Sample data
            dist: Distribution to test against ('norm', 'exp', 'logistic')
            
        Returns:
            Dictionary with test results
        """
        from scipy.stats import anderson
        
        result = anderson(sample, dist=dist)
        
        return {
            'statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_levels': result.significance_level,
            'reject_null': result.statistic > result.critical_values[2]  # 5% level
        }
    
    # ==================== Correlation ====================
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        from scipy.stats import pearsonr
        
        r, p_value = pearsonr(x, y)
        return r, p_value
    
    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Spearman rank correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        from scipy.stats import spearmanr
        
        rho, p_value = spearmanr(x, y)
        return rho, p_value
    
    @staticmethod
    def kendall_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kendall's tau correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (tau, p-value)
        """
        from scipy.stats import kendalltau
        
        tau, p_value = kendalltau(x, y)
        return tau, p_value
    
    @staticmethod
    def point_biserial_correlation(continuous: np.ndarray, 
                                   binary: np.ndarray) -> Tuple[float, float]:
        """
        Compute point-biserial correlation.
        
        Args:
            continuous: Continuous variable
            binary: Binary variable (0/1)
            
        Returns:
            Tuple of (correlation, p-value)
        """
        from scipy.stats import pointbiserialr
        
        r_pb, p_value = pointbiserialr(continuous, binary)
        return r_pb, p_value
    
    @staticmethod
    def partial_correlation(x: np.ndarray, y: np.ndarray, 
                           z: np.ndarray) -> float:
        """
        Compute partial correlation between x and y controlling for z.
        
        Args:
            x: First variable
            y: Second variable
            z: Control variable(s)
            
        Returns:
            Partial correlation coefficient
        """
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        
        # Regress x on z
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression().fit(z, x)
        resid_x = x - reg_x.predict(z)
        
        # Regress y on z
        reg_y = LinearRegression().fit(z, y)
        resid_y = y - reg_y.predict(z)
        
        # Correlation of residuals
        r, _ = Statistics.pearson_correlation(resid_x, resid_y)
        return r
    
    @staticmethod
    def correlation_matrix(data: np.ndarray, 
                          method: str = 'pearson') -> np.ndarray:
        """
        Compute correlation matrix for multivariate data.
        
        Args:
            data: Data matrix (n_samples x n_features)
            method: 'pearson', 'spearman', or 'kendall'
            
        Returns:
            Correlation matrix (n_features x n_features)
        """
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if method == 'pearson':
                    r, _ = Statistics.pearson_correlation(data[:, i], data[:, j])
                elif method == 'spearman':
                    r, _ = Statistics.spearman_correlation(data[:, i], data[:, j])
                elif method == 'kendall':
                    r, _ = Statistics.kendall_tau(data[:, i], data[:, j])
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
        
        return corr_matrix
    
    # ==================== Effect Size ====================
    
    @staticmethod
    def cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Cohen's d
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def eta_squared(ss_between: float, ss_total: float) -> float:
        """
        Compute eta-squared effect size for ANOVA.
        
        Args:
            ss_between: Sum of squares between groups
            ss_total: Total sum of squares
            
        Returns:
            Eta-squared
        """
        return ss_between / ss_total
    
    @staticmethod
    def odds_ratio(a: int, b: int, c: int, d: int) -> float:
        """
        Compute odds ratio from 2x2 contingency table.
        
        Args:
            a: Cell (1,1) - exposed with outcome
            b: Cell (1,2) - exposed without outcome
            c: Cell (2,1) - unexposed with outcome
            d: Cell (2,2) - unexposed without outcome
            
        Returns:
            Odds ratio
        """
        if b * c == 0:
            return np.inf
        return (a * d) / (b * c)
    
    @staticmethod
    def relative_risk(a: int, b: int, c: int, d: int) -> float:
        """
        Compute relative risk from 2x2 contingency table.
        
        Args:
            a: Cell (1,1) - exposed with outcome
            b: Cell (1,2) - exposed without outcome
            c: Cell (2,1) - unexposed with outcome
            d: Cell (2,2) - unexposed without outcome
            
        Returns:
            Relative risk
        """
        risk_exposed = a / (a + b) if (a + b) > 0 else 0
        risk_unexposed = c / (c + d) if (c + d) > 0 else 0
        
        if risk_unexposed == 0:
            return np.inf
        return risk_exposed / risk_unexposed
    
    # ==================== Resampling Methods ====================
    
    @staticmethod
    def bootstrap(data: np.ndarray, statistic: callable, 
                  n_bootstrap: int = 1000, seed: int = None) -> np.ndarray:
        """
        Perform bootstrap resampling.
        
        Args:
            data: Original sample
            statistic: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
            
        Returns:
            Bootstrap distribution of statistic
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(data)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            bootstrap_stats[i] = statistic(bootstrap_sample)
        
        return bootstrap_stats
    
    @staticmethod
    def permutation_test(sample1: np.ndarray, sample2: np.ndarray,
                        statistic: callable, n_permutations: int = 1000,
                        alternative: str = "two-sided",
                        seed: int = None) -> HypothesisTestResult:
        """
        Permutation test (randomization test).
        
        Args:
            sample1: First sample
            sample2: Second sample
            statistic: Function to compute test statistic
            n_permutations: Number of permutations
            alternative: 'two-sided', 'greater', or 'less'
            seed: Random seed
            
        Returns:
            HypothesisTestResult
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Observed statistic
        observed = statistic(sample1) - statistic(sample2)
        
        # Combine samples
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        # Permutation distribution
        perm_stats = np.zeros(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_sample1 = combined[:n1]
            perm_sample2 = combined[n1:]
            perm_stats[i] = statistic(perm_sample1) - statistic(perm_sample2)
        
        # P-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        elif alternative == "greater":
            p_value = np.mean(perm_stats >= observed)
        else:  # less
            p_value = np.mean(perm_stats <= observed)
        
        return HypothesisTestResult(
            statistic=observed,
            p_value=p_value,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def cross_validation_score(model, X: np.ndarray, y: np.ndarray,
                               cv: int = 5, scoring: str = 'accuracy',
                               seed: int = None) -> Tuple[float, float]:
        """
        K-fold cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            seed: Random seed
            
        Returns:
            Tuple of (mean score, std score)
        """
        from sklearn.model_selection import cross_val_score
        
        if seed is not None:
            np.random.seed(seed)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores), np.std(scores)
    
    # ==================== Outlier Detection ====================
    
    @staticmethod
    def detect_outliers_iqr(data: np.ndarray, 
                           multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using IQR method.
        
        Args:
            data: Input data
            multiplier: IQR multiplier (1.5 for mild, 3.0 for extreme)
            
        Returns:
            Boolean mask of outliers
        """
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, 
                               threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using z-score method.
        
        Args:
            data: Input data
            threshold: Z-score threshold
            
        Returns:
            Boolean mask of outliers
        """
        z_scores = np.abs(Statistics.z_scores(data))
        return z_scores > threshold
    
    @staticmethod
    def detect_outliers_mad(data: np.ndarray, 
                            threshold: float = 3.5) -> np.ndarray:
        """
        Detect outliers using Median Absolute Deviation (MAD).
        
        Args:
            data: Input data
            threshold: Modified z-score threshold
            
        Returns:
            Boolean mask of outliers
        """
        robust_z = np.abs(Statistics.robust_z_scores(data, method='mad'))
        return robust_z > threshold
    
    @staticmethod
    def remove_outliers(data: np.ndarray, method: str = 'iqr',
                       **kwargs) -> np.ndarray:
        """
        Remove outliers from data.
        
        Args:
            data: Input data
            method: 'iqr', 'zscore', or 'mad'
            **kwargs: Additional arguments for outlier detection
            
        Returns:
            Data with outliers removed
        """
        if method == 'iqr':
            mask = Statistics.detect_outliers_iqr(data, **kwargs)
        elif method == 'zscore':
            mask = Statistics.detect_outliers_zscore(data, **kwargs)
        elif method == 'mad':
            mask = Statistics.detect_outliers_mad(data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data[~mask]
