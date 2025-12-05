## Logistic regression with L2 penalty Model Setup
import numpy as np
from optimizers_logreg import fit_gd_fixed, fit_gd_armijo, armijo_backtracking, fit_nlcg_prp, fit_bfgs

# Sigmoid link function
def sigmoid(z):
    """Compute the Sigmoid activation function with clipping for safety."""
    # clip inputs to avoid overflow in exp for very large magnitude values
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z)) # standard logistic transform returning values in (0,1)

# Logistic regression loss (objective) function and its gradient
def logreg_loss_and_grad(w, X, y, lamda):
    """
    Compute L2-regularized logistic regression loss and its gradient.
        
    Inputs
        w: parameter (coefficient) vector (d,)
        X: matrix with intercept (n,d)
        y: target labels in {0,1} (n,)
        lamda : L2 penalty strength (scalar)
    
    Returns
        loss: scalar logistic loss with L2 penalty
        grad: gradient vector (d,)
    """
        
    # N of observations
    n = X.shape[0]
    # Linear predictor z_i = X_i^T w for all row at once
    z = X @ w # linear combination of inputs and weights
    # Predicted probabilities (p_i = sigmoid(z_i))
    p = sigmoid(z)
    
    # Avoid log(0) numerical issues by clipping p slightly
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # Average negative log-likelihood (data loss)
    # = mean(-y*log(p) - (1-y)*log(1-p))
    data_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    # L2 penalty term (from Gussian prior with variance of sigma^2 = 1/lambda)
    reg_loss = 0.5 * lamda * (w @ w)
    
    # Total objective function
    loss = data_loss + reg_loss # Total loss = data loss + regularization
    
    # Gradient of data loss (analytical derivative)
    grad = (X.T @ (p - y)) / n + lamda * w # derivative of loss w.r.t. weights; gradient vector same size as w
    
    return loss, grad

def logreg_loss_and_grad_no_intercept_penalty(w, X, y, lamda):
    """L2-regularized logistic regression without intercept regularization."""
    # N of observations
    n = X.shape[0]
    # Linear predictor z_i = X_i^T w for all row at once
    z = X @ w # linear combination of inputs and weights
    # Predicted probabilities (p_i = sigmoid(z_i))
    p = sigmoid(z)
    
    # Avoid log(0) numerical issues by clipping p slightly
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # Data loss
    data_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    # L2 penalty -> intercept(w[0]) excluded
    reg_loss = 0.5 * lamda * (w[1:] @ w[1:]) # w[0] excluded
    
    loss = data_loss + reg_loss
    
    # Gradient -> intercept no penalized
    grad = (X.T @ (p - y)) / n
    grad[1:] += lamda * w[1:] # w[0] same; w[1:] only penalized
    
    return loss, grad

def add_intercept(X):
    """Adds a column of ones to the feature matrix for the bias term."""
    # A column of ones to X to model an intercept
    return np.c_[np.ones(X.shape[0]), X]

def predict_prob(w, X):
    """Compute predicted probabilities for logistic regression."""
    # Probability of class 1 given features (X)
    return sigmoid(X @ w)

def predict_label(w, X, thresh=0.5):
    """Compute binary predicted labels based on threshold."""
    # Convert probs to hard labels using a threshold (0.5)
    return (predict_prob(w, X) >= thresh).astype(int)

def accuracy(w, X, y, thresh=0.5):
    # Fraction or proportion of correct predictions (thr = 0.5)
    return np.mean(predict_label(w, X, thresh) == y)

# Wrapper function to select optimizer
def fit_logreg(X, y, lam=1.0, optimizer="gd_armijo", step=1e-2, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    Main training function that routes to the specific optimization algorithm.
    
    Inputs
        X: array (n, d) without intercept column. This function will add one.
        y: array in {0,1}
        lam: L2 strength
        optimizer: "gd", "gd_armijo", "cg", "bfgs"
        step: used only for fixed-step GD
        alpha0: initial step for Armijo-based methods
    Returns
        (w, info) where info has status, iters, history
    """
    # Convert X to array and add intercept term
    Xb = add_intercept(np.asarray(X, dtype=float))
    yb = np.asarray(y, dtype=float).reshape(-1)
    
    # Wrap loss and gradient function
    fg = lambda w: logreg_loss_and_grad(w, Xb, yb, lam)
    
    # Initialize weights with zeros
    w0 = np.zeros(Xb.shape[1], dtype=float)
    
    # Select and run optimizer
    if optimizer == 'gd':
        w, info = fit_gd_fixed(fg, w0, step=step, tol=tol, max_iter=max_iter)
    elif optimizer == 'gd_armijo':
        # Run Nonlinear Conjugate Gradient
        w, info = fit_gd_armijo(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == 'cg':
        # Run Nonlinear Conjugate Gradient
        w, info = fit_nlcg_prp(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == 'bfgs':
        # Run Quasi-Newton BFGS
        w, info = fit_bfgs(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')
        
    return w, info

def fit_logreg_no_intercept_penalty(X, y, lam=1.0, optimizer="gd_armijo", step=1e-2, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    Main training function that routes to the specific optimization algorithm.
    
    Inputs
        X: array (n, d) without intercept column. This function will add one.
        y: array in {0,1}
        lam: L2 strength
        optimizer: "gd", "gd_armijo", "cg", "bfgs"
        step: used only for fixed-step GD
        alpha0: initial step for Armijo-based methods
    Returns
        (w, info) where info has status, iters, history
    """
    # Convert X to array and add intercept term
    Xb = add_intercept(np.asarray(X, dtype=float))
    yb = np.asarray(y, dtype=float).reshape(-1)
    
    # Wrap loss and gradient function
    fg = lambda w: logreg_loss_and_grad_no_intercept_penalty(w, Xb, yb, lam)
    
    # Initialize weights with zeros
    w0 = np.zeros(Xb.shape[1], dtype=float)
    
    # Select and run optimizer
    if optimizer == 'gd':
        w, info = fit_gd_fixed(fg, w0, step=step, tol=tol, max_iter=max_iter)
    elif optimizer == 'gd_armijo':
        # Run Nonlinear Conjugate Gradient
        w, info = fit_gd_armijo(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == 'cg':
        # Run Nonlinear Conjugate Gradient
        w, info = fit_nlcg_prp(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == 'bfgs':
        # Run Quasi-Newton BFGS
        w, info = fit_bfgs(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')
        
    return w, info

# helper to build arrays from DataFrames
def df_to_Xy(df, y_col):
    """
    Converts a DataFrame to (X,y). No scaling. No intercept.
    y is mapped to {0,1} if it looks like {-1,1}.
    """
    X = df.drop(columns=[y_col]).to_numpy(dtype=float)
    y = df[y_col].to_numpy()
    # map {-1,1} to {0,1} if needed
    if set(np.unique(y)).issubset({-1, 1}):
        y = ((y + 1) // 2).astype(int)
    else:
        y = y.astype(int)
    return X, y