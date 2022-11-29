"""This file contains utilities used in generating the paper figures."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def top_down_theory_bound(p: float, q: float, ell_star: float, n: int, d_star: float, error: float) -> float:
    """Function to calculate the theoretical sufficient k given experimental 
    parameters for the top-down oracle-based algorithm.

    Args:
        p: lambda, the mutation rate
        q: the state distribution collision rate
        ell_star: the triplet ingroup/outgroup LCA distance error parameter
        n: the number of leaves
        d_star: the depth error parameter
        error: zeta, the probability error parameter

    Returns:
        The sufficient k, the number of characters
    """

    val = 1/(2*p) * np.log((1-q)/q) + 1
    if val < 0:
        delta_star = (1 - q) + q * np.exp(-2 * p)
    elif val >= 0 and val <= d_star:
        delta_star = 2 * np.exp(-p) * np.sqrt(q * (1 - q))
    else:
        delta_star = np.exp(-p * d_star) * (1-q) + q * np.exp(-p * (2 - d_star))
    delta_star = 0.6 * delta_star

    return (96 * np.log(n) + 32 * np.log(1/error)) * (ell_star + (1 - np.exp(-p)) * q)/(ell_star**2 * 0.6 * delta_star * p * (1 - q + q * np.exp(-2 * p))) 

def bottom_up_theory_bound(p: float, q: float, ell_star: float, n: int, c: float, error: float) -> float:
    """Function to calculate the theoretical sufficient k given experimental 
    parameters for the bottom-up algorithm.

    Args:
        p: lambda, the mutation rate
        q: the state distribution collision rate
        ell_star: the triplet ingroup/outgroup LCA distance error parameter
        n: the number of leaves
        c: constant equal to (maximum edge length)^2 / minimum edge length
        error: zeta, the probability error parameter

    Returns:
        The sufficient k, the number of characters
    """
    if q > 3/(16 * (1 - np.exp(-p * (c * ell_star) ** (1/2)))):
        return np.nan
    
    gamma = 2 * (c * ell_star)**(1/2)
    C = gamma * np.exp(-p) + p * gamma ** 2 * q
    
    beta = p * q * max(1, c)
    if beta > 1/(1 + C + 2 * (np.exp(-p * ell_star) + p * gamma * q) ** 2):
        return np.nan
    
    denom = p * np.exp(-p) * ell_star * (1 - beta * (1 + C)) * (1 - beta * (1 + C)  - 2 * beta * (np.exp(-p * ell_star) + p * gamma * q) ** 2)

    return (20 * np.log(n) + 10 * np.log(1 / error))/denom

def missing_theory_bound(p: float, q: float, ell_star: float, n: int, d_star: float, m: float, error: float) -> float:
    """Function to calculate the theoretical sufficient k given experimental 
    parameters for the top-down oracle-based algorithm with (stochastic only)
    missing data.

    Args:
        p: lambda, the mutation rate
        q: the state distribution collision rate
        ell_star: the triplet ingroup/outgroup LCA distance error parameter
        n: the number of leaves
        d_star: the depth error parameter
        m: the proportion of stochastic missing data
        error: zeta, the probability error parameter

    Returns:
        The sufficient k, the number of characters
    """
    val = 1/(2*p) * np.log((1-q)/q) + 1
    if val < 0:
        delta_star = (1 - q) + q * np.exp(-2 * p)
    elif val >= 0 and val <= d_star:
        delta_star = 2 * np.exp(-p) * np.sqrt(q * (1 - q))
    else:
        delta_star = np.exp(-p * d_star) * (1-q) + q * np.exp(-p * (2 - d_star))
    delta_star = 0.6 * delta_star  

    return (96 * np.log(n) + 32 * np.log(1/error)) * (ell_star + p * q + np.exp(p * d_star) * d_star)/((1-m)**2 * ell_star**2 * 0.6 * delta_star * p * (1 - q + q * np.exp(-2 * p))) 

def logistic_spline(series: pd.Series) -> int:
    """Function to perform a logistic spline on simulation results for k.

    Given the proportion of trees that are correct (meeting the reconstruction 
    criteria) for each value of k, fits a logistic regression to reduce noise.
    Then, the minimum k that passes 0.9 proportion correct is reported.
    """
    x, y = series[2], series[3]

    # If there are not at least 2 data points, return nan
    if len(x) < 2:
        return np.nan

    # Fit the logistic regression model
    X = sm.add_constant(x)
    model = sm.Logit(y, X)
    try:
        results = model.fit(method = "Newton", disp = 0)

    # If the model fails to fit, likely because all proportions are 0,
    # return nan
    except sm.tools.sm_exceptions.PerfectSeparationError:
        return np.nan

    # Get ks that are predicted to reach 0.9 proportion, up to a large maximum
    max_x = 20000
    vals = sm.add_constant(list(range(max_x)))
    preds = results.predict(vals)
    x_over = [i for i in range(len(preds)) if preds[i] >= 0.9]
    
    # If no values in the searched range of ks reach 0.9 proportion, return nan
    if len(x_over) == 0:
        return np.nan
    else:
        # Find the smallest k that achieves 0.9 proportion
        final_val = min(x_over)
        # If this final value is greater than a size bound, return nan
        if final_val <= 4096:
            return final_val
        else:
            return np.nan

def find_lower_bound(series: pd.Series) -> int:
    x, y = series[2], series[3]
    X = sm.add_constant(x)
    
    if len(x) < 2:
        return np.nan
    
    model = sm.Logit(y, X)
    try:
        results = model.fit(method = "Newton", disp = 0)
    except sm.tools.sm_exceptions.PerfectSeparationError:
        return np.nan
    max_x = 20000
    
    vals = sm.add_constant(list(range(max_x)))
    preds = results.predict(vals)
    x_over = [i for i in range(len(preds)) if preds[i] >= 0.9]
    
    # estimate confidence interval for predicted probabilities
    cov = results.cov_params()
    gradient = (preds * (1 - preds) * vals.T).T # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    c = 1.96 # multiplier for confidence interval
    upper = np.maximum(0, np.minimum(1, preds + std_errors * c))
    
    x_over = [i for i in range(len(upper)) if upper[i] >= 0.9]
    
    if len(x_over) == 0:
        return np.nan
    else:
        return min(x_over)

def find_upper_bound(series: pd.Series) -> int:
    x, y = series[2], series[3]
    X = sm.add_constant(x)
    
    if len(x) < 2:
        return np.nan
    
    model = sm.Logit(y, X)
    try:
        results = model.fit(method = "Newton", disp = 0)
    except sm.tools.sm_exceptions.PerfectSeparationError:
        return np.nan
    max_x = 20000
    
    vals = sm.add_constant(list(range(max_x)))
    preds = results.predict(vals)
    x_over = [i for i in range(len(preds)) if preds[i] >= 0.9]
    
    # estimate confidence interval for predicted probabilities
    cov = results.cov_params()
    gradient = (preds * (1 - preds) * vals.T).T # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    c = 1.96 # multiplier for confidence interval
    lower = np.maximum(0, np.minimum(1, preds - std_errors * c))
    
    x_over = [i for i in range(len(lower)) if lower[i] >= 0.9]
    
    if len(x_over) == 0:
        return np.nan
    else:
        return min(x_over)