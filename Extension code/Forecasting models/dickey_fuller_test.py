from statsmodels.tsa.stattools import adfuller
import pandas as pd

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test on a time series.
    
    Parameters:
    series (pd.Series): The time series to test.
    
    Returns:
    dict: A dictionary with the test statistic, p-value, and critical values.
    """
    adf_dict = {}
    for col in series.columns:
        result_trend = adfuller(series[col], autolag='AIC', regression='ct')
        result_no_trend = adfuller(series[col], autolag='AIC', regression='c')
        adf_dict[col] = {
            'ADF Statistic': result_trend[0],
            'p-value': result_trend[1],
            'Lag order': result_trend[2],
            'ADF Statistic (no trend)': result_no_trend[0],
            'p-value (no trend)': result_no_trend[1],
            'Lag order (no trend)': result_no_trend[2],
        }
    return adf_dict

fwd_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
in_sample_end = pd.to_datetime("1989-12-01")
fwd_rates_insample = fwd_rates[fwd_rates['Date'] <= in_sample_end]

in_sample_results_dict = adf_test(fwd_rates_insample)
full_results_dict = adf_test(fwd_rates)

in_sample_results_df = pd.DataFrame.from_dict(in_sample_results_dict, orient='index')
in_sample_results_df.to_excel("data-folder/!Data for forecasting/adf_test_results.xlsx", index_label='Forward Rate')
full_results_df = pd.DataFrame.from_dict(full_results_dict, orient='index')
full_results_df.to_excel("data-folder/!Data for forecasting/adf_test_results_full.xlsx", index_label='Forward Rate')