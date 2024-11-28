import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from rich import print

def generate_data_ridge(n: int = 100, 
                        p: int = 3, 
                        beta: np.ndarray = np.array([0.89, 1.5,-3.2]), 
                        eps: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_
    --------------------------------------------------------------
    Lecture 2 implementantion: Ridge regression.
    --------------------------------------------------------------
    This function simulate a toy dataset to fit a ridge regression

    Args:
        n (int, optional): _description_. Defaults to 100.=> number of obs
        p (int, optional): _description_. Defaults to 3.=> no: covariates
        beta (np.ndarray, optional): _description_. Defaults to np.array([0.89, 1.5,-3.2]).
        eps (float, optional): _description_. Defaults to 1.0.=> variance

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    # Set seed for reproducibility
    np.random.seed(29102)
    X = np.random.rand(n, p - 1)  # Generate predictors
    X = np.hstack([np.ones((n, 1)), X])  # Add intercept
    # Simulate the response using the typical LM 
    y = X @ beta + np.random.normal(0, eps, size = n) # shape (n,)
    return X, y


def ridge_estimate(X: np.ndarray, 
                   y: np.ndarray,
                   lamb: float)->np.ndarray:
    """_summary_
     --------------------------------------------------------------
    Lecture 2 implementantion: Ridge regression.
    --------------------------------------------------------------
    This function fit a ridge regression to a toy dataset

    Args:
        X: (np.ndarray): _description_: Data matrix : shape (n, p)
        y: (np.ndarray): _description_: Response: shape (n,)
        lamb (float): _description_: A scalar quantity-->regularization parameter

    Returns:
        np.ndarray: _description_
    """
    p = X.shape[1] # shape ==> (p,)
    I = np.eye(p) # shape ==> (p, p) --> diaganol matrix
    # Do not regularize the intercept
    I[0, 0] = 0 
    # Compute Beta-hat ridge 
    beta_ridge = np.linalg.inv(X.T @ X + lamb * I) @ (X.T @ y) # shape ==> (p,)
    return beta_ridge


def evaluate_ridge(n_sim: int = 1000, 
                   n: int = 100, 
                   p: int =3, 
                   beta: np.ndarray = np.array([0.89, 1.5,-3.2]), 
                   eps: float = 1.0, 
                   lamb = 0.89)->Tuple[pd.DataFrame, np.ndarray]:
    """_summary_
    --------------------------------------------------------------
    Lecture 2 implementantion: Ridge regression.
    --------------------------------------------------------------
    This function evaluate the ridge estimates
    
    Args:
        n_sim (int, optional): _description_. Defaults to 1000.=> total iterations
        n (int, optional): _description_. Defaults to 100.=> number of obs
        p (int, optional): _description_. Defaults to 3.=> number of parameters
        beta (np.ndarray, optional): _description_. Defaults to np.array([0.89, 1.5,-3.2]).
        eps (float, optional): _description_. Defaults to 1.0.=> error variance
        lamb (float, optional): _description_. Defaults to 0.89.=> regularization term

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: _description_
    """
    # A container
    beta_estimates: List[np.ndarray] = []
    # repeat 1000 times
    for _ in range(n_sim):
        
        # Fetch the data
        X, y = generate_data_ridge(n = n, 
                                   p = p, 
                                   beta = beta, 
                                   eps = eps)
        # Fit the ridge regression
        beta_hat = ridge_estimate(X, y, lamb)
        # collect the parameters
        beta_estimates.append(beta_hat)
        
    beta_estimates = np.array(beta_estimates)
    
    # Compute Bias, Variance, and MSE
    bias = np.mean(beta_estimates, axis=0) - beta
    variance = np.var(beta_estimates, axis=0)
    mse = bias**2 + variance
    # Pack the data
    results = pd.DataFrame({
        'True Beta': beta,
        'Bias': bias,
        'Variance': variance,
        'MSE': mse})
    
    return results, beta_estimates

# Plot parameter estimates vs lambda
def plot_ridge_path(X, y, beta_true, lambdas):
    p = X.shape[1]
    beta_paths = np.zeros((len(lambdas), p))
    for i, lamb in enumerate(lambdas):
        beta_paths[i, :] = ridge_estimate(X, y, lamb)
    
    #plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    for j in range(p):
        plt.plot(lambdas, beta_paths[:, j], label=f"Beta {j} (True: {beta_true[j]})")
    plt.axhline(0, color="gold", linestyle="--", linewidth=0.8)
    plt.title("Ridge Regression: Parameter Estimates vs Regularization")
    plt.xlabel("Regularization Parameter (Lambda)")
    plt.ylabel("Parameter Estimates")
    plt.legend()
    plt.grid(False)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    n, p = 100, 3
    beta_true = np.array([0.89, 1.5,-3.2])
    X, y = generate_data_ridge(n=n, 
                               p=p, 
                               beta=beta_true,
                               eps=1)

    # Evaluate ridge regression for a single lambda
    lamb = 10
    results, beta_estimates = evaluate_ridge(n_sim=1000, n=n, p=p,
                                             beta=beta_true, 
                                             eps=1, lamb=lamb)
    print(f"\n{160 * '*'}\n")
    print(f"\n Evaluation of Ridge Regression with λ = {lamb}\n")
    print(results)
    print(f"\n{160 * '*'}\n")
    
    _, beta_estimates = evaluate_ridge()
    print(f"\n>>>> estimates: {beta_estimates.mean(axis = 0)}, True values = {beta_true}\n")

    # Plot Ridge Path
    lambdas = np.logspace(-3, 3, 100)  # Regularization parameter range
    plot_ridge_path(X, y, beta_true, lambdas)