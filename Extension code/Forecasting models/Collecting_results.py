from itertools import product
import pandas as pd
from Roos import r2_oos
from bayesian_shrinkage import bayesian_shrinkage

# Load the data
#fwd_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression/FWD_reg.xlsx")
#Macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression/Macro_reg.xlsx")
#rf_fwd_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Random forest preds/FWD_rf.xlsx")
#rf_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Random forest preds/Macro_rf.xlsx")
en_fwd_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/ElasticNet preds/FWD_en.xlsx")
en_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/ElasticNet preds/Macro_en.xlsx")
preds = {"FWD": en_fwd_preds, "Macro": en_macro_preds}
         #, "FWD": rf_fwd_preds, "Macro": rf_macro_preds}

benchmark_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/benchmark.xlsx")
realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")

models = ["Regression"] #, "Random Forest"]  # We assume only one model for now.
priors = [0, 0.25, 0.5, 0.75]
inputs = ["FWD", "Macro"]

# Define the horizons you want to compute Roos for.
horizons = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]

results_list = []
for model in models:  # Loop through each model explicitly
    for input_type, pred_df in preds.items():
        for prior in priors:
            for horizon in horizons:
                # Extract actual values, benchmark predictions, and the model predictions for the given horizon.
                actual = realized[horizon].values
                benchmark = benchmark_preds[horizon].values
                predicted = pred_df[horizon].values

                # Calculate Roos for raw predictions.
                roos_original = r2_oos(actual, predicted, benchmark)

                # Apply Bayesian shrinkage to get new predictions.
                shrunk_pred = bayesian_shrinkage(benchmark, predicted, prior_weight=prior)
                # Calculate Roos for shrinkage predictions.
                roos_shrunk = r2_oos(actual, shrunk_pred, benchmark)

                # Save the results along with the configuration.
                results_list.append({
                    "Model": model,  # Save the current model explicitly
                    "Input": input_type,
                    "Prior": prior,
                    "Horizon": horizon,
                    "Roos_Original": roos_original,
                    "Roos_Shrunk": roos_shrunk
                })

# Convert list of dicts into a DataFrame and save to Excel.
results = pd.DataFrame(results_list)
print(results)
results.to_excel("Extension code/Forecasting models/Saved preds/roos.xlsx", index=False)


