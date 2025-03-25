from sklearn.decomposition import PCA
#from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
forward_rates = pd.read_excel("data-folder/Extracted_fwd_rates.xlsx", parse_dates=True).iloc[:-1, :]
excess_returns = pd.read_excel("data-folder/Extracted_excess_returns.xlsx", parse_dates=True)
# Take the first difference of excess returns
excess_returns = excess_returns.iloc[:, 1:].diff().dropna()
forward_rates = forward_rates.iloc[:, 1:].diff().dropna()

#print(excess_returns.head())
#print(forward_rates.head())


# Standardise forward_rates before applying PCA
scaler = StandardScaler()
forward_rates.iloc[:, 1:] = scaler.fit_transform(forward_rates.iloc[:, 1:])

# Apply PCA to forward rates
pca = PCA()
pca.fit(forward_rates.iloc[:, 1:])

# Scree plot: Variance explained by each principal component

explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.grid()
plt.show()


# Apply PCA to forward rates
# Use only the first principal component
forward_rates_pca = pca.transform(forward_rates.iloc[:, 1:])[:, :1]
PC1 = pd.DataFrame(forward_rates_pca, columns=["PC1"])


# Regress excess returns on the first principal component
'''
reg = LinearRegression()
reg.fit(PC1, excess_returns)
'''

# Regress excess returns on the first principal component using statsmodels
PC1_with_const = sm.add_constant(PC1)  # Add an intercept term
model = sm.OLS(excess_returns['36 m'], PC1_with_const)  # Ordinary Least Squares regression
results = model.fit()

# Print the summary of the fitted regression model
print(results.summary())


# DOESN'T WORK!!!
