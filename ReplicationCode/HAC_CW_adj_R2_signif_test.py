import statsmodels.api as sm
from scipy.stats import t as tstat
import numpy as np

def get_CW_adjusted_R2_signif(y_true, y_forecast, bechmark):

    y_true = np.asarray(y_true)
    y_forecast = np.asarray(y_forecast)

    # Format benchmark
    benchmark_forecast = bechmark
    benchmark_forecast[np.isnan(y_forecast)] = np.nan  # Align NaNs

    # Forecast errors
    e_bench = y_true - benchmark_forecast
    e_model = y_true - y_forecast
    diff_bm = benchmark_forecast - y_forecast

    # MSPE-adjusted error difference
    f = (e_bench ** 2) - ((e_model ** 2) - (diff_bm ** 2))

    # Drop NaNs
    f = f[~np.isnan(f)]

    # Regress f on a constant using HAC robust standard errors
    x = np.ones_like(f)
    model = sm.OLS(f, x)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    tstat_val = results.tvalues[0]
    p_value = 1 - tstat.cdf(tstat_val, df=results.nobs - 1)

    return tstat_val, p_value