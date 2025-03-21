import pandas as pd


def clean_data(yield_data_path, macro_data_path, yield_date_col, macro_date_col):
    """
    Cleans and aligns yield and macroeconomic data based on their date ranges.

    Parameters
    ----------
    yield_data_path : str
        File path to the yield data Excel file.
    macro_data_path : str
        File path to the macroeconomic data CSV file.
    yield_date_col : str
        Column name for the date in the yield data.
    macro_date_col : str
        Column name for the date in the macroeconomic data.

    Returns
    -------
    None
        Saves the aligned datasets to separate Excel files.
    """
    # Read the input files
    Yields = pd.read_excel(yield_data_path, skiprows=8)
    MacroData = pd.read_csv(macro_data_path, skiprows=0)
    MacroData = MacroData.drop(0)

    # Convert the Date columns to datetime format
    Yields[yield_date_col] = pd.to_datetime(Yields[yield_date_col].astype(str), format="%Y%m")
    MacroData[macro_date_col] = pd.to_datetime(MacroData[macro_date_col])

    # Align the datasets based on the date range
    start_date = max(Yields[yield_date_col].min(), MacroData[macro_date_col].min())
    end_date = min(Yields[yield_date_col].max(), MacroData[macro_date_col].max())

    Yields_aligned = Yields[(Yields[yield_date_col] >= start_date) & (Yields[yield_date_col] <= end_date)]
    MacroData_aligned = MacroData[(MacroData[macro_date_col] >= start_date) & (MacroData[macro_date_col] <= end_date)]

    # Save the aligned datasets to separate Excel files
    Yields_aligned.to_excel("data-folder\Aligned_Yields.xlsx", index=False)
    MacroData_aligned.to_excel("data-folder\Aligned_VintageMacroData.xlsx", index=False)

    print("Aligned datasets have been saved as 'Aligned_Yields.xlsx' and 'Aligned_VintageMacroData.xlsx'.")
