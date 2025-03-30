import pandas as pd

# Load Excel and clean headers
df = pd.read_excel(r"data-folder\Cleaned data\Yields+Final\Aligned_Yields.xlsx", index_col=0)

# Clean column names and convert them to integers
df.columns = df.columns.str.strip().str.replace(" m", "", regex=False).astype(int)

# Convert percentage points to decimals
df = df / 100.0

# Get list of column indices (maturities in months)
cols = list(df.columns)  # e.g. [1, 2, ..., 120]

# Prepare output DataFrame
df_excess_returns = pd.DataFrame(index=df.index)

# Loop over maturities > 12 months
for i in range(len(cols)):
    n = cols[i]
    if n <= 12:
        continue  # skip short maturities

    idx_n = i                # index of maturity n
    idx_n_minus_12 = i - 12  # index of maturity n-12

    # Get the yield columns directly by index
    y_t_n = df.iloc[:, idx_n]
    y_t_n_minus_12 = df.iloc[:, idx_n_minus_12]
    y_t_12 = df.iloc[:, cols.index(12)]  # short rate

    # Compute log prices
    p_t_n = -(n / 12) * y_t_n
    p_tplus12_nminus12 = -((n - 12) / 12) * y_t_n_minus_12.shift(-12)       

    # Compute holding period return and excess return
    holding_return = p_tplus12_nminus12 - p_t_n
    excess_return = holding_return - y_t_12

    # Save
    df_excess_returns[n] = excess_return

# Drop rows with NaN from shift
df_excess_returns = df_excess_returns.dropna()

# Save full excess return matrix
output_path = r"data-folder\Cleaned data\Yields+Final\Excess_returns.xlsx"
df_excess_returns.to_excel(output_path)
print(f"Excess return results saved to: {output_path}")
print(df_excess_returns.head())

# === NEW SECTION: Filtered subset and save ===

# Define target maturities
target_maturities = [24, 36, 48, 60, 84, 120]

# Filter and rename columns
df_extracted = df_excess_returns[target_maturities]
df_extracted.columns = [f"{n} m" for n in target_maturities]

# Save filtered version
extracted_output = r"data-folder\Extracted_excess_returns.xlsx"
df_extracted.to_excel(extracted_output)
print(f"✅ Extracted excess returns saved to: {extracted_output}")
print(df_extracted.head())
