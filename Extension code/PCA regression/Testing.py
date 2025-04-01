import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos

def split_data_by_date(excess_returns, forward_rates, split_date, end_date, macro_data=None):
    """
    Splits excess_returns, forward_rates, and optionally macro_data into in-sample and out-of-sample sets.
    """
    def split(df):
        return df[df["Date"] < split_date], df[(df["Date"] >= split_date) & (df["Date"] <= end_date)]
    
    data = {
        "excess_returns_insample": None,
        "excess_returns_oos": None,
        "forward_rates_insample": None,
        "forward_rates_oos": None,
        "macro_data_insample": None,
        "macro_data_oos": None,
    }
    
    data["excess_returns_insample"], data["excess_returns_oos"] = split(excess_returns)
    data["forward_rates_insample"], data["forward_rates_oos"] = split(forward_rates)
    
    if macro_data is not None:
        data["macro_data_insample"], data["macro_data_oos"] = split(macro_data)
    
    return {k: v.drop(columns="Date") if v is not None else None for k, v in data.items()}

def iterative_pca_regression(excess_returns_insample, forward_rates_insample,
                             excess_returns_oos, forward_rates_oos,
                             macro_data_insample=None, macro_data_oos=None):
    """
    Performs iterative PCA regression, updating the model with each new out-of-sample observation.
    """
    pred_values = []
    
    # Fit PCA on forward rates
    n_components = 3
    pca = PCA(n_components).fit(forward_rates_insample)
    pcs_insample = pca.transform(forward_rates_insample)
    
    # Fit PCA on macro data if provided
    if macro_data_insample is not None:
        macro_scaler = StandardScaler().fit(macro_data_insample)
        macro_pcs_insample = PCA(n_components=8).fit_transform(macro_scaler.transform(macro_data_insample))
    else:
        macro_pcs_insample = None
    
    X_insample = np.hstack([pcs_insample, macro_pcs_insample]) if macro_pcs_insample is not None else pcs_insample
    y_insample = excess_returns_insample
    
    model = LinearRegression().fit(X_insample, y_insample)
    
    # Iterative updating
    for i in range(len(excess_returns_oos)):
        test_fwd_pcs = pca.transform(forward_rates_oos.iloc[[i]])
        
        if macro_data_oos is not None:
            test_macro_pcs = PCA(n_components=8).fit_transform(
                macro_scaler.transform(macro_data_oos.iloc[[i]])
            )
            test_X = np.hstack([test_fwd_pcs, test_macro_pcs])
        else:
            test_X = test_fwd_pcs
        
        pred_values.append(model.predict(test_X)[0, 0])
        
        # Update dataset
        excess_returns_insample = pd.concat([excess_returns_insample, excess_returns_oos.iloc[[i]]])
        forward_rates_insample = pd.concat([forward_rates_insample, forward_rates_oos.iloc[[i]]])
        
        if macro_data_oos is not None:
            macro_data_insample = pd.concat([macro_data_insample, macro_data_oos.iloc[[i]]])
        
        # Refit PCA and model
        pca = PCA(n_components).fit(forward_rates_insample)
        pcs_insample = pca.transform(forward_rates_insample)
        
        if macro_data_insample is not None:
            macro_scaler = StandardScaler().fit(macro_data_insample)
            macro_pcs_insample = PCA(n_components=8).fit_transform(macro_scaler.transform(macro_data_insample))
            X_insample = np.hstack([pcs_insample, macro_pcs_insample])
        else:
            X_insample = pcs_insample
        
        model.fit(X_insample, excess_returns_insample)
    
    return pd.Series(pred_values, index=excess_returns_oos.index)

if __name__ == "__main__":
    # Load data
    forward_rates = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx")
    macro_data = pd.read_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx")
    
    # Define out-of-sample period
    start_oos, end_oos = "1990-01-01", "2018-12-01"
    
    # Split data
    data = split_data_by_date(excess_returns, forward_rates, start_oos, end_oos, macro_data=macro_data)
    
    # Columns to predict
    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    predictions = {}
    
    for column in columns_to_predict:
        print(f"Running iterative PCA regression for {column}")
        predictions[column] = iterative_pca_regression(
            data["excess_returns_insample"][column],
            data["forward_rates_insample"],
            data["excess_returns_oos"][column],
            data["forward_rates_oos"],
            macro_data_insample=data["macro_data_insample"],
            macro_data_oos=data["macro_data_oos"]
        )
    
    # Compute benchmark predictions and out-of-sample R²
    benchmark_predictions = compute_benchmark_prediction(
        data["excess_returns_insample"], data["excess_returns_oos"])
    
    for column in columns_to_predict:
        r2 = r2_oos(data["excess_returns_oos"][column], predictions[column], benchmark_predictions[column])
        print(f"Out-of-sample R² for {column}: {r2:.4f}")
