def bayesian_shrinkage(benchmark, model_predictions, prior_weight=0.5):
    for i in range(len(model_predictions)):
        # Compute the average of the benchmark and model predictions
        avg_val = prior_weight * benchmark.iloc[i] + (1 - prior_weight) * model_predictions.iloc[i]
        # Update the model predictions with the average value
        model_predictions.iloc[i] = avg_val

    return model_predictions