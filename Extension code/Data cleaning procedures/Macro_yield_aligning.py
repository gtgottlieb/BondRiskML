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


def extract_and_save_data(dataframes, columns_to_extract, output_folder="data-folder", file_names=None):
    """
    Extracts specified columns and rows from a list of dataframes and saves them to Excel files.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        List of dataframes to process.
    columns_to_extract : list of str
        List of column names to extract from each dataframe.
    start_date : str
        The starting date (inclusive) for row extraction in 'YYYY-MM-DD' format.
    output_folder : str, optional
        Folder where the extracted dataframes will be saved. Default is "data-folder".
    file_names : list of str, optional
        List of file names for saving the extracted dataframes. If None, default names will be used.

    Returns
    -------
    None
        Saves the extracted dataframes to Excel files in the specified folder.
    """
    # Extract the specified columns from the dataframes
    extracted_dataframes = []
    for df in dataframes:
        extracted_dataframes.append(df[columns_to_extract])
    

    # Set default file names if not provided
    if file_names is None:
        file_names = [f"Extracted_Data_{i+1}.xlsx" for i in range(len(dataframes))]

    # Save all extracted dataframes to Excel files in the output folder
    for df, file_name in zip(extracted_dataframes, file_names):
        df.to_excel(f"{output_folder}/{file_name}", index=False)

    print("Extracted dataframes have been saved to the specified folder.")


def process_and_extract_columns(dataframe, maturities, output_folder="data-folder", file_name="Extracted_Data.xlsx"):
    """
    Processes a dataframe by standardizing column names, extracting specific maturity columns,
    and converting maturities from months to years. Retains the date column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process and extract columns from.
    maturities : list of int
        List of maturities (in months) to extract columns for.
    output_folder : str, optional
        Folder where the extracted dataframe will be saved. Default is "data-folder".
    file_name : str, optional
        File name for saving the extracted dataframe. Default is "Extracted_Data.xlsx".

    Returns
    -------
    None
        Saves the processed and extracted dataframe to an Excel file.
    """
    # Standardize column names by stripping leading/trailing spaces
    dataframe.columns = dataframe.columns.str.strip()

    # Extract the columns for the specified maturities
    columns_to_extract = [f"{i} m" for i in maturities]
    extracted_data = dataframe[["Date"] + columns_to_extract]  # Include the 'Date' column

    # Rename the maturity columns to years
    years = [f"{i // 12} y" for i in maturities]
    extracted_data.columns = ["Date"] + years

    # Save the extracted data to an Excel file
    extracted_data.to_excel(f"{output_folder}/{file_name}", index=False)

    print(f"Extracted data has been saved to '{output_folder}/{file_name}'.")


if __name__ == "__main__":
    # Extracting maturities of 1 to 10 years
    aligned_yields = pd.read_excel("data-folder/Cleaned data/Yields+Final/Aligned_Yields.xlsx")
    maturities = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]

    process_and_extract_columns(aligned_yields, maturities, file_name="Aligned_Yields_Extracted.xlsx")
    

    '''
    # Load excel files
    Excess_returns_df = pd.read_excel("data-folder/Cleaned data/Yields+Final/Excess_Returns.xlsx")
    Forward_rates_df = pd.read_excel("data-folder/Cleaned data/Yields+Final/Forward_Rates.xlsx")

    # Define the maturities to extract
    maturities = [24, 36, 48, 60, 84, 120]

    # Process and extract columns for each dataframe
    process_and_extract_columns(Excess_returns_df, maturities, file_name="Excess_Returns_Extracted.xlsx")
    process_and_extract_columns(Forward_rates_df, maturities, file_name="Forward_Rates_Extracted.xlsx")
    '''

    