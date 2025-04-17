import numpy as np
import scipy.stats as stats


def calculate_confidence_interval(data, confidence=0.95):
    """
    Compute confidence intervals for the mean and variance of each numerical column in a Pandas DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical columns.
        confidence (float): Confidence level (default 0.95 for 95%).

    Returns:
        dict: Confidence intervals for mean and variance of each column.
    """
    results = {}
    alpha = 1 - confidence
    data = data.dropna().values  # Remove NaN values
    n = len(data)

    if n < 2:
        return

    # Sample mean and sample variance
    mean = np.mean(data)
    variance = np.var(data, ddof=1)  # Sample variance

    # Mean CI
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)  # t critical value
    mean_margin = t_critical * (np.sqrt(variance) / np.sqrt(n))
    mean_ci = (mean - mean_margin, mean + mean_margin)

    # Variance CI
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n-1)  # Lower chi-square critical value
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n-1)  # Upper chi-square critical value

    var_ci_lower = (n - 1) * variance / chi2_upper
    var_ci_upper = (n - 1) * variance / chi2_lower if chi2_lower > 0 else np.nan  # Prevent division by zero

    variance_ci = (var_ci_lower, var_ci_upper)

    # Store results
    results = {
        "mean": mean,
        "mean_CI": mean_ci,
        "variance": variance,
        "variance_CI": variance_ci
    }
    return results