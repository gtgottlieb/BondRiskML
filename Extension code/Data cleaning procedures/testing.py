import pandas as pd
import numpy as np
from statsmodels.iolib.summary2 import summary_col
from statsmodels.formula.api import ols
from statsmodels.stats.sandwich_covariance import cov_hac
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_forward_rates(yield_df):
    """
    Calculate forward rates and excess returns as per Liu & Wu (2020).
    """
    # Ensure yields are sorted by maturity (e.g., 1y, 2y, ..., 5y)
    maturities = sorted(yield_df.columns[1:], key=lambda x: int(x.split()[0]))
    yield_df = yield_df[['Date'] + maturities]
    
    # Log prices: logP_t(n) = -n * y_t(n)
    logP = -yield_df[maturities] * np.arange(1, len(maturities)+1)  # Assuming maturities start at 1y
    
    # Forward rates: f_t(n) = logP_t(n-1) - logP_t(n)
    forward_rates = pd.DataFrame(index=yield_df.index)
    for i, n in enumerate(maturities[1:], start=1):  # Skip 1y
        prev_n = maturities[i-1]
        forward_rates[n] = logP[prev_n] - logP[n]
    forward_rates.insert(0, '1y', yield_df['1y'])  # f_t(1) = y_t(1)
    
    # Excess returns: rx_{t+1}(n) = [logP_{t+1}(n-1) - logP_t(n)] - y_t(1)
    excess_returns = pd.DataFrame(index=yield_df.index)
    for i, n in enumerate(maturities[1:], start=1):  # For 2y to 5y
        prev_n = maturities[i-1]
        logP_shifted = logP[prev_n].shift(-12)  # Shift monthly data by 12 for 1-year ahead
        rx = (logP_shifted - logP[n]) - yield_df['1y']
        excess_returns[n] = rx
    excess_returns.dropna(inplace=True)
    
    return forward_rates, excess_returns

def run_regression(X, y, newey_lags=18):
    """Run OLS with Newey-West standard errors."""
    model = ols('y ~ X', data=pd.DataFrame({'y': y, 'X': X})).fit()
    cov = cov_hac(model, nlags=newey_lags)
    return model, cov

# Load and preprocess data
yield_data = pd.read_excel("Yields.xlsx", parse_dates=['Date'])
yield_data.set_index('Date', inplace=True)

# Compute forward rates and excess returns
forward_rates, excess_returns = get_forward_rates(yield_data.reset_index())

# Average excess returns for 2y-5y
excess_returns['rx_avg'] = excess_returns[['2y', '3y', '4y', '5y']].mean(axis=1)

# Split into original (1964-2003) and extended (1964-2019) samples
mask_orig = (forward_rates.index >= '1964-01-01') & (forward_rates.index <= '2003-12-31')
mask_ext = (forward_rates.index >= '1964-01-01') & (forward_rates.index <= '2019-12-31')

# Standardize forward rates for PCA
scaler = StandardScaler()
X_orig = scaler.fit_transform(forward_rates.loc[mask_orig, ['1y', '2y', '3y', '4y', '5y']])
X_ext = scaler.transform(forward_rates.loc[mask_ext, ['1y', '2y', '3y', '4y', '5y']])

# Perform PCA
pca = PCA(n_components=5)
pca_orig = pca.fit_transform(X_orig)
pca_ext = pca.transform(X_ext)

# Regression 1: Forward rates
X_fwd_orig = forward_rates.loc[mask_orig, ['1y', '2y', '3y', '4y', '5y']]
y_orig = excess_returns.loc[mask_orig, 'rx_avg']
model_fwd_orig, cov_fwd_orig = run_regression(X_fwd_orig, y_orig)

# Regression 2: Principal components
X_pca_orig = pd.DataFrame(pca_orig, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=X_fwd_orig.index)
model_pca_orig, cov_pca_orig = run_regression(X_pca_orig, y_orig)

# Print results
print("Original Sample (1964-2003)")
print("Forward Rates Regression:")
print(model_fwd_orig.summary())
print("\nPrincipal Components Regression:")
print(model_pca_orig.summary())

# Repeat for extended sample...


