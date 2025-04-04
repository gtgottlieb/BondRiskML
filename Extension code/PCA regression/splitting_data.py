import pandas as pd


def split_data_by_date(excess_returns: pd.DataFrame,
                       forward_rates: pd.DataFrame,
                       split_date, end_date,
                       macro_data: pd.DataFrame = None) -> dict:
    """
    Splits excess returns, forward rates, and optionally macro data into 
    in-sample and out-of-sample sets.

    Args:
        excess_returns (pd.DataFrame): DataFrame with a 'Date' column.
        forward_rates (pd.DataFrame): DataFrame with a 'Date' column.
        split_date: Start date for the out-of-sample period.
        end_date: End date for the out-of-sample period.
        macro_data (pd.DataFrame, optional): DataFrame with a 'Date' column.

    Returns:
        dict: Dictionary containing in-sample and out-of-sample datasets.
    """
    in_er = excess_returns.loc[excess_returns["Date"] < split_date].copy()
    out_er = excess_returns.loc[(excess_returns["Date"] >= split_date) & 
                                (excess_returns["Date"] <= end_date)].copy()

    in_fr = forward_rates.loc[forward_rates["Date"] < split_date].copy()
    out_fr = forward_rates.loc[(forward_rates["Date"] >= split_date) & 
                               (forward_rates["Date"] <= end_date)].copy()

    if macro_data is not None:
        in_macro = macro_data.loc[macro_data["Date"] < split_date].copy()
        out_macro = macro_data.loc[(macro_data["Date"] >= split_date) & 
                                   (macro_data["Date"] <= end_date)].copy()
    else:
        in_macro, out_macro = None, None

    return {
        "excess_returns_in": in_er,
        "excess_returns_out": out_er,
        "forward_rates_in": in_fr,
        "forward_rates_out": out_fr,
        "macro_data_in": in_macro,
        "macro_data_out": out_macro,
    }