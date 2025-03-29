import pandas as pd
import statsmodels.api as sm

# Import data
Y = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Extracted_excess_returns.xlsx', index_col=0)
X = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Extracted_fwd_rates.xlsx', index_col=0)
Y['avg'] = Y.mean(axis=1)

results = {}

# Run OLS for each maturity
for maturity in Y.columns:
    X_maturity = X.copy() # Using all maturities of fwd rates for each excess return
    # X_maturity = X[maturity] # Using corresponding maturity of fwd rates for
    Y_maturity = Y[maturity]

    if maturity == '120 m': # 10-year maturities start later
        start_date = '1972-08-01'
        X_maturity = X_maturity.loc[start_date:]
        Y_maturity = Y_maturity.loc[start_date:]

    model = sm.OLS(Y_maturity, X_maturity).fit()
    
    results[maturity] = model.summary()

# Print results 
for maturity, result in results.items():
    print(f"Results for maturity {maturity}:\n")
    print(result)
    print("\n" + "-"*50 + "\n")