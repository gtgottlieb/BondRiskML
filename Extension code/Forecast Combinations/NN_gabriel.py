import numpy as np
import pandas as pd

from pca_regression import split_data_by_date
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def iterative_nn_regression(er_in: pd.DataFrame,
                            fr_in: pd.DataFrame,
                            er_out: pd.DataFrame,
                            fr_out: pd.DataFrame,
                            macro_in: pd.DataFrame = None,
                            macro_out: pd.DataFrame = None) -> pd.Series:
    """
    Performs iterative neural network regression using an expanding window.
    For each out-of-sample period, trains an MLPRegressor on in-sample data up to 11 periods before.
    """
    predictions = []
    if macro_in is not None and macro_out is not None:
        X_in = np.hstack([fr_in.values, macro_in.values])
    else:
        X_in = fr_in.values
    y_in = er_in.values.flatten()
    
    for i in range(len(er_out)):
        if macro_in is not None and macro_out is not None:
            X_test = np.hstack([fr_out.iloc[[i]].values, macro_out.iloc[[i]].values])
        else:
            X_test = fr_out.iloc[[i]].values
        if i <= 11 or len(X_in) <= 11:
            pred = np.mean(y_in)
        else:
            X_train = X_in[:-11]  # use training data excluding the most recent 11 observations
            y_train = y_in[:-11]
            nn_model = MLPRegressor(hidden_layer_sizes=(3,),
                                    activation='relu',
                                    solver='adam',
                                    random_state=42,
                                    max_iter=500)
            nn_model.fit(X_train, y_train)
            pred = nn_model.predict(X_test)[0]
        predictions.append(pred)
        # Expand the training data with the newly observed out-of-sample point.
        new_feat = fr_out.iloc[[i]].values
        if macro_in is not None and macro_out is not None:
            new_feat = np.hstack([new_feat, macro_out.iloc[[i]].values])
        X_in = np.vstack([X_in, new_feat])
        y_in = np.append(y_in, er_out.iloc[i].values)
    return pd.Series(predictions, index=er_out.index)

if __name__ == "__main__":
    # Load data
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 

    # Define out-of-sample period.
    start_oos = pd.to_datetime("1990-01-01")
    end_oos = pd.to_datetime("2023-12-01")

    # Convert 'Date' columns to datetime.
    for df in [forward_rates, excess_returns, macro_data]:
        df["Date"] = pd.to_datetime(df["Date"])
        
    # Split data into in-sample and out-of-sample.
    data_split = split_data_by_date(excess_returns, forward_rates, start_oos, end_oos, macro_data=None)
    
    # Drop the 'Date' column.
    for key in data_split:
        if data_split[key] is not None:
            data_split[key] = data_split[key].drop(columns="Date")
    

    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    realized = er_out.copy() # For computing IR
    #er_out.to_excel("data-folder/realized_xr.xlsx", index=False)
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]

    # Calculate benchmark predictions using in-sample and out-of-sample excess returns
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    predictions = {}

    # Calculate and print OOS R² scores for each maturity column for both prediction sets")
    for col in columns_to_predict:
        print(f"Running iterative NN regression for column: {col}")
        er_in_col = er_in[[col]].copy()
        er_out_col = er_out[[col]].copy()

        pred = iterative_nn_regression(
            er_in_col,
            fr_in.copy(),
            er_out_col,
            fr_out.copy(),
            macro_in = None,   # or supply macro data if available
            macro_out = None   # or supply macro data if available
        )
        predictions[col] = pred

        r2_value = r2_oos(er_out[col], predictions[col], benchmark_preds[col])
        print(f"Out-of-sample R2 for {col}: {r2_value}")


    # Compute benchmark predictions.
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)
    # Save all predictions from all maturity columns as a DataFrame
    