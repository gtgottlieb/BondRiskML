import pandas as pd


def align_data(yield_df, macro_df, yield_save_name, macro_save_name, yield_date_col='Date', macro_date_col='sasdate'):
    """
    Cleans and aligns yield and macroeconomic data based on their date ranges.

    Parameters
    ----------
    yield_df : pd.DataFrame
        DataFrame containing the yield data.
    macro_df : pd.DataFrame
        DataFrame containing the macroeconomic data.
    yield_date_col : str
        Column name for the date in the yield data.
    macro_date_col : str
        Column name for the date in the macroeconomic data.
    yield_save_name : str
        File name to save the aligned yield data.
    macro_save_name : str
        File name to save the aligned macroeconomic data.

    Returns
    -------
    None
        Saves the aligned datasets to separate Excel files.
    """
    # Convert the Date columns to datetime format
    yield_df[yield_date_col] = pd.to_datetime(yield_df[yield_date_col].astype(str), format="%Y%m")
    macro_df[macro_date_col] = pd.to_datetime(macro_df[macro_date_col])

    # Align the datasets based on the date range
    start_date = max(yield_df[yield_date_col].min(), macro_df[macro_date_col].min())
    end_date = min(yield_df[yield_date_col].max(), macro_df[macro_date_col].max())

    yield_aligned = yield_df[(yield_df[yield_date_col] >= start_date) & (yield_df[yield_date_col] <= end_date)]
    macro_aligned = macro_df[(macro_df[macro_date_col] >= start_date) & (macro_df[macro_date_col] <= end_date)]

    # Save the aligned datasets to separate Excel files
    yield_aligned.to_excel(f"data-folder\\{yield_save_name}", index=False)
    macro_aligned.to_excel(f"data-folder\\{macro_save_name}", index=False)

    print(f"Aligned datasets have been saved as '{yield_save_name}' and '{macro_save_name}'.")
