import numpy as np

def r2_oos(actual, predicted, benchmark):
    """
    Computes the out-of-sample R^2 (R^2_oos) as per Campbell and Thompson (2008).

    Parameters:
    - actual: np.array of actual values x_{T, t:t+12}^{(n)}
    - predicted: np.array of predicted values \hat{x}_{T, t:t+12}^{(n)}(M)
    - benchmark: np.array of benchmark predictions \bar{x}_{T, t:t+12}^{(n)}

    Returns:
    - R^2_oos value
    """
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - benchmark) ** 2)
    
    return 1 - (numerator / denominator)