import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

#macro = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx").iloc[:, 1:]
fwd_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx").iloc[:, 1:]

# Standardize the data before PCA
'''
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(macro)
'''

# Perform PCA
pca = PCA()
pca.fit(fwd_rates)
explained_variance = pca.explained_variance_ratio_


# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.xticks(np.arange(1, len(cumulative_explained_variance) + 1))
plt.show()

'''
# Create a scree plot
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.xticks(np.arange(1, len(explained_variance) + 1, 10))
plt.show()
'''