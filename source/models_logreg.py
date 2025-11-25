## Logistic regression with L2 penalty Model Setup
import numpy as np
from optimizers_logreg import fit_gd_fixed, fit_gd_armijo, armijo_backtracking, fit_nlcg_prp, fit_bfgs

# sigmoid link function
def sigmoid(z):
    # clip inputs to avoid overflow in exp for very large magnitude values
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z)) # standard logistic transform returning values in (0,1)

# logistic regression loss (objective) function and its gradient
def logreg_loss_and_grad(w, X, y, lamda):
    """
    Inputs
    w: parameter (coefficient) vector (d,)
    X: design matrix with intercept if you added it (n,d)
    y: target labels in {0,1} (n,)
    lamda : L2 penalty strength (scalar)

    Returns
      loss: scalar logistic loss with L2 penalty
      grad: gradient vector (d,)
    """
    
    # n of observations
    n = X.shape[0]
    # linear predictor z_i = X_i^T w for all row at once
    z = X @ w
    # predicted probabilities (p_i = sigmoid(z_i))
    p = sigmoid(z)

    # avoid log(0) numerical issues by clipping p slightly
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # average negative log-likelihood
    # = mean(-y*log(p) - (1-y)*log(1-p))
    data_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # L2 penalty term (from Gussian prior with variance of sigma^2 = 1/lambda)
    reg_loss = 0.5 * lamda * (w @ w)

    # total objective function
    loss = data_loss + reg_loss

    grad = (X.T @ (p - y)) / n + lamda * w
    return loss, grad

def add_intercept(X):
    # a column of ones to X to model an intercept
    return np.c_[np.ones(X.shape[0]), X]

def predict_prob(w, X):
    # probability of class 1 given features (X)
    return sigmoid(X @ w)

def predict_label(w, X, thresh=0.5):
    # convert probs to hard labels using a threshold (0.5)
    return (predict_prob(w, X) >= thresh).astype(int)

def accuracy(w, X, y, thresh=0.5):
    # fraction or proportion of correct predictions (thr = 0.5)
    return np.mean(predict_label(w, X, thresh) == y)

# ---------- high level training wrapper ----------
def fit_logreg(X, y, lam=1.0, optimizer="gd_armijo",
               step=1e-2, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    X: array (n, d) without intercept column. This function will add one.
    y: array in {0,1}
    lam: L2 strength
    optimizer: "gd", "gd_armijo", "cg", "bfgs"
    step: used only for fixed-step GD
    alpha0: initial step for Armijo-based methods
    """
    # ensure array and add intercept
    Xb = add_intercept(np.asarray(X, dtype=float))
    yb = np.asarray(y, dtype=float).reshape(-1)

    # lambda wrapper for
    # objective and gradient wrapper f(w), g(w)
    fg = lambda w: logreg_loss_and_grad(w, Xb, yb, lam)

    # initialize weights; start at zeros
    w0 = np.zeros(Xb.shape[1], dtype=float)

    # choose optimizer
    if optimizer == "gd":
        w, info = fit_gd_fixed(fg, w0, step=step, tol=tol, max_iter=max_iter)
    elif optimizer == "gd_armijo":
        w, info = fit_gd_armijo(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == "cg":
        # Nonlinear Conjugate Gradient
        w, info = fit_nlcg_prp(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == "bfgs":
        # Quasi-Newton BFGS
        w, info = fit_bfgs(fg, w0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    return w, info

# ---------- tiny helper to build arrays from your DataFrames ----------
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