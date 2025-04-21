def bayesian_shrinkage(benchmark, model_predictions, prior_weight=0.5):
    # Create a new series with the weighted average.
    return prior_weight * benchmark + (1 - prior_weight) * model_predictions