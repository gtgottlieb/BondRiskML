import os
import zipfile
import pandas as pd
from data_cleaning import clean_data

def extract_last_rows_from_vintages(vintages_folder=r"c:\Users\Jan\OneDrive - Erasmus University Rotterdam\Desktop\BondRiskML\Vintages", 
                                    output_file=r"c:\Users\Jan\OneDrive - Erasmus University Rotterdam\Desktop\BondRiskML\data-folder\Raw data\vintages.csv"):
    """
    Extracts the last rows from all CSV files within ZIP archives in a specified folder 
    and saves them to a single CSV file.

    Args:
        vintages_folder (str): Path to the folder containing ZIP files.
        output_file (str): Path to save the resulting CSV file.
    """
    # Create a DataFrame to store the last rows
    last_rows = []

    # Iterate through all zip files in the folder
    for zip_filename in os.listdir(vintages_folder):
        if zip_filename.endswith(".zip"):
            zip_path = os.path.join(vintages_folder, zip_filename)
            
            # Open the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all .csv files to a temporary location
                for file in zip_ref.namelist():
                    if file.endswith(".csv"):
                        # Read the .csv file directly from the zip
                        with zip_ref.open(file) as csv_file:
                            try:
                                # Read the CSV file into a DataFrame
                                df = pd.read_csv(csv_file)
                                
                                # Check if the DataFrame is empty
                                if df.empty:
                                    continue
                                
                                # Get the last row of the DataFrame
                                last_row = df.iloc[-1]
                                last_rows.append(last_row)
                            except Exception:
                                pass

    # Combine all last rows into a single DataFrame
    if last_rows:
        result_df = pd.DataFrame(last_rows)
        # Save the result to a new .csv file
        result_df.to_csv(output_file, index=False)




if __name__ == "__main__":
   yield_df = pd.read_excel("data-folder/Raw data/LW_monthly.xlsx", skiprows=8)
   vintages_df = pd.read_csv("data-folder/Raw data/vintages.csv")
   #print(yield_df.columns)
   clean_data(yield_df, vintages_df, macro_save_name="vintage_aligned.xlsx", yield_save_name="lw_monthly_aligned.xlsx")

