import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from rich import print 

def generate_data(n: int =1000, 
                  beta: int = np.array([2, -3, 5]),
                  eps: float =1)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ---------------------------------------
    ST 6105 Lecture 2-implementation (WLOS). 
    ---------------------------------------
    
    _summary_
    
    This function simulate the data to fit a Weghted Least Square regression. 
    The key idea is to make sure the variance acrross the subjects differs.

    Args:
        n (int, optional): _description_. Defaults to 100.--> Total number of subjects
        beta (np.ndarray, optional): _description_. Defaults to np.array([2, -3, 5]). 
        Which is a column vector of Reg coefficients
        eps (float, optional): _description_. Defaults to 1. The error term

    Returns:
        _type_: _description_ A tuple of consists of covariates, response and errors
    
    """
    # Set seed for reproducibility
    np.random.seed(12180)
    
    # Simulate data matrix (Asume we have X1 & X2) i.e, p=2
    X = np.random.rand(n, len(beta) - 1) # shape (n, p-1 = 2)
    
    # Add the intercept column of ones
    X = np.hstack([np.ones((n, 1)), X]) # shape == (n, p)
    
    # Simulating heteroscedasticity (here, we use X1 but any can be chosen or both)
    wts = 1 + X[:, 1]  # shape (n,)
    
    # For numerical stability we avoid extreme weights
    wts = np.clip(wts, 1e-6, 1e6)
      
    # To stabilize the errors we clip to the min of 1e-6
    scale = np.clip(eps * wts, 1e-6, np.max(eps * wts))
    
    # Sample a heteroscedastic var from a normal distribution
    errors = np.random.normal(0, scale, size =n ) # shape ==> (n,)
    
    # Obtain the response use a LM relation 
    y = X @ beta + errors # shape (n,)
    
    return X, y, wts

# def wls_estimate(X: np.ndarray,
#                  y:np.ndarray, 
#                  weights: np.ndarray) -> np.ndarray:
#     """
#      ---------------------------------------
#     ST 6105 Lecture 2-implementation (WLOS). 
#     ---------------------------------------
    
#     _summary_
    
#     This function implement the Weighted Least Square estimation
#     --------------------------------

#     Args:
#         X (np.ndarray): _description_: Covariate data with shape [n, p]
#         y (np.ndarray): _description_: Response of shape (n, )
#         weights (np.ndarray): _description_: The weight matrix to introduce heteroscedasticity: shape [n,]
#     Returns:
#         np.ndarray: _description_ Vector of Parameter estimates with shape: [p,]
#     """
#     W = np.diag(weights) # We only  need the variances
#     # Compute the WLS estimator
#     beta_hat = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y) # (p, n) @ (n,) @ (n, p) @ (p,n) @ (n,),(n,)=>(p,)
#     return beta_hat # shape ==> (p,)

def compute_wls(X, y, wts)->np.ndarray:
    """
    _summary_
    
    This function implement the Weighted Least Square estimation
    --------------------------------
    ST 6105 Lecture 2-implementation. 
    ---------------------------------

    Args:
        X (np.ndarray): _description_: Covariate data with shape [n, p]
        y (np.ndarray): _description_: Response of shape (n, )
        weights (np.ndarray): _description_: The weight matrix to introduce heteroscedasticity: shape [n,]
    Returns:
        np.ndarray: _description_ Vector of Parameter estimates with shape: [p,]
    
    """
    # We need to obtain the diaganol matrix for the weight
    W = np.diag(wts)  # shape ==> (n, n)
    
    # Ensure that the resulting matrix is invertible
    try:
        beta_hat = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)# shape ==> (p,)
    except np.linalg.LinAlgError:
        print("Matrix inversion failed: Possibly singular matrix.")
        print(f"Rank of X.T @ W @ X: {np.linalg.matrix_rank(X.T @ W @ X)}")
        raise
    return beta_hat # shape ==> (p,)


def evaluate_wls(n_sim: int = 1000, 
                 n: int = 100, 
                 beta: np.ndarray = np.array([2, -3, 5]), 
                 eps: float = 1.0)->Tuple[pd.DataFrame, np.ndarray]:
    
    """_summary_
    To evaluate the performance of WLS estimates we compute the
    Bias, Variance, and the MSE.
    
    --------------------------------
    ST 6105 Lecture 2-implementation. 
    ---------------------------------
    
    Args:
        n_sim (int, optional): _description_. Defaults to 1000.
        n (int, optional): _description_. Defaults to 100.
        beta (np.ndarray, optional): _description_. Defaults to np.array([2, -3, 5]).
        eps (float, optional): _description_. Defaults to 1.0.

    Returns:
        Tuple: _description_: results
    """
    beta_estimates: List = [] # Container to hold all WLS betas for 1000 iterations

    for _ in range(n_sim):
        # Generate data
        X, y, eps = generate_data(n = n, beta = beta, eps = eps)
        # introduce heteroscedasticity [We assumed only x1 involves]
        weights = 1 / (1 + X[:, 1]) # shape ==> [n,]  

        # WLS estimation
        beta_hat = compute_wls(X, y, weights) # shape ==> (p,)
        beta_estimates.append(beta_hat) 

    beta_estimates = np.array(beta_estimates) # shape ==> (1000, p)
    
    # Compute Bias, Variance, and MSE
    bias = np.mean(beta_estimates, axis=0) - beta # shape ==> (p,)
    variance = np.var(beta_estimates, axis=0) # shape [p,]
    mse = bias**2 + variance # shape ==> [p,]

    results = pd.DataFrame({
        'True Beta': beta,
        'Bias': bias,
        'Variance': variance,
        'MSE': mse
    })
    return results, beta_estimates


# Ploting the distribution of estimates for visualization

results, beta_estimates = evaluate_wls()
plt.figure(figsize=(10, 6))
for i, beta_name in enumerate(["Intercept", "Beta 1", "Beta 2"]):
    plt.hist(beta_estimates[:, i], bins=30, alpha=0.7, label=f"{beta_name} estimated")
    plt.axvline(results["True Beta"][i], color="red", linestyle="--", label=f"True {beta_name}" if i == 0 else "")
plt.title("Distribution of WLS Estimates")
plt.xlabel("Estimate")
plt.ylabel("Frequency")
plt.legend()
plt.show()

if __name__ == "__main__":
    # X, y, eps = generate_data()
    # print(f"\n {160 * '*'}\n")
    # print(f"X shape: {X.shape}, y shape: {y.shape}, epsilon shape: {eps.shape}\n")
    # print(f"\n {160 * '*'}\n")
    # a = 1 / (1 + X[:, 1])
    # print("a shape: ", a.shape)
    
    results, beta_estimates = evaluate_wls()

    # Display results
    print("Evaluation of Weighted Least Squares (WLS):")
    print(f"True betas: [2, -3, 5]: WLE of betas:{beta_estimates.mean(axis = 0)}\n")
    print("Evaluation of Weighted Least Squares (WLS):")
    print(results)