import numpy as np
import pandas as pd
import yfinance as yf
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Fetch S&P 500 returns data from Yahoo Finance
ticker = '^GSPC' # S&P 500 index ticker
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')
data['Returns'] = data['Adj Close'].pct_change().dropna()
# Use the returns data for BOCP with Dirichlet Priors
returns = data['Returns'].dropna().values

# Estimate the empirical hazard rate using kernel density estimation
def estimate_hazard_rate_kernel(data):
    '''
    Estimate the empirical hazard rate using Kernel Density Estimation.
    Parameters:
        - data: The time series data (e.g., S&amp;P 500 returns).
    Returns:
        - hazard_rate: The estimated hazard rate.

    
    '''
    changes = np.abs(np.diff(data))
    kde = gaussian_kde(changes, bw_method='silverman')
    hazard_rate = kde.evaluate(np.abs(data[:-1]))
    hazard_rate /= hazard_rate.max()
    return hazard_rate

# BOCPD with Dirichlet Process Prior using PyMC3
def bocpd_dp_pymc3(data):
    '''
    Apply BOCPD with a Dirichlet Process Prior using PyMC3.
    
    Parameters:
        - data: The time series data (e.g., S&amp;P 500 returns).
    Returns:
        - trace: The MCMC trace from the PyMC3 model.
        - hazard_rate: The estimated hazard rate from KDE.
    '''

    with pm.Model() as model:
        # Dirichlet Process for Change Point Detection
        alpha = pm.Gamma('alpha', alpha=1, beta=1)
        k = pm.Categorical('k', 
                           p=pm.Dirichlet('p',a=alpha*np.ones(len(data))), 
                           shape=len(data))
        # Priors for the t-distribution parameters
        mu = pm.Normal('mu', mu=0, sigma=1, shape=len(data))
        sigma = pm.InverseGamma('sigma', alpha=2, beta=1, shape=len(data))

        nu = pm.Exponential('nu', 1/30, shape=len(data)) # Degrees of freedom for t-distribution
        # Likelihood function: t-distribution for returns or volatility
        likelihood = pm.StudentT('likelihood', nu=nu[k], mu=mu[k], sigma=sigma[k], observed=data)
        # Estimate the hazard rate using kernel density estimation
        hazard_rate = estimate_hazard_rate_kernel(data)
        # Perform MCMC sampling to get the posterior distribution
        trace = pm.sample(draws=500, chains=2, return_inferencedata=True, cores=4, progressbar=True)
        # trace = pmjax.sample_numpyro_nuts(draws=1000, chains=2)
    
    return trace, hazard_rate


if __name__ == '__main__':

    print(f'No. points in test: {len(returns)}')
    # Run BOCPD on the S&P 500 returns using the Dirichlet Process Prior
    trace, hazard_rate = bocpd_dp_pymc3(returns[::10])