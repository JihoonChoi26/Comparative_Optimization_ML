### Model Setup for Hard-Margin SVM
import numpy as np

from optimizers_logreg import fit_gd_fixed, fit_gd_armijo, armijo_backtracking
from models_logreg import add_intercept

def svm_loss_and_grad(theta, X, y, C = 1000, loss = "squared_hinge"):
    """
    X: (n, d+1) feature matrix that already includes a column of ones (intercept) at the last column
    y: in {-1, +1}
    theta: [w, b] stacked with b as the last entry
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    
    # n samples, d+1 columns (with intercept)
    n, d1 = X.shape

    # define parameters into w (first d) and b (last)
    w, b = theta[:-1], theta[-1]

    # scores (s_i = w^T x_i + b)
    s = X @ theta                     
    
    # signed margins (m_i = y_i * s_i)
    m = y * s                       

    if loss == "squared_hinge":
        # v_i = max(0, 1 - m_i), violations of margin 1
        v = np.maximum(0, 1 - m)

        # objective: 0.5 ||w||^2 + C * mean(v^2)
        loss_val = 0.5 * (w @ w) + C * np.mean(v**2)

        # grad_w = w - (2C/n) sum_i v_i y_i x_i
        coef = (2 * C / n) * v * y          # length n
        grad_w = w - (X[:, :-1].T @ coef)     # (d,)
        
        # grad_b = -(2C/n) sum_i v_i y_i = - sum_i coef_i
        grad_b = - np.sum(coef)

        # stack back into a single vector [grad_w, grad_b] 
        grad = np.r_[grad_w, grad_b]

    elif loss == "hinge": # Non-smooth; return a subgradient
        # same violations
        v = np.maximum(0.0, 1.0 - m)

        # objective: 0.5 ||w||^2 + C * mean(v)
        loss_val = 0.5 * (w @ w) + C * np.mean(v)

        # Active set indicator (I_i = 1 if v_i > 0)
        I = (v > 0.0).astype(float)         

        # subgrad_w = w - (C/n) sum_i I_i y_i x_i
        coef = (C / n) * I * y
        subgrad_w = w - (X[:, :-1].T @ coef)

        # subgrad_b = -(C/n) sum_i I_i y_i
        subgrad_b = - np.sum(coef)

        # stack back into a single vector
        grad = np.r_[subgrad_w, subgrad_b]

    else:
        raise ValueError("loss must be 'squared_hinge' or 'hinge'")

    return loss_val, grad

def fit_subgradient(fg, theta0, alpha0=1, tol=1e-6, max_iter=10000):
    '''
    fg: callable that maps theta -> (f(theta), g) where g is a subgradient
    theta0: initial parameter vector
    alpha0: base step size used in alpha_k = alpha0 / sqrt(k + 1)
    tol: stop when ||g||_2 <= tol
    max_iter: maximum number of iterations
    '''
    # current iterate
    theta = theta0.copy()
    
    # log
    hist = []
    
    for k in range(max_iter):
        # evaluate objective and (sub)gradient at current iterate
        f, g = fg(theta) # g is a (sub)gradient

        # norm of g
        gnorm = np.linalg.norm(g)

        # log progress
        hist.append((k, f, gnorm))

        # stop if small enough subgraident norm
        if gnorm <= tol:
            return theta, {"status": 0, "iters": k, "history": hist}
        
        # diminishing step
        alpha = alpha0 / np.sqrt(k + 1.0)
        
        # subgraident step
        theta = theta - alpha * g
    
    # return 1 if not converged within max_iter
    return theta, {"status": 1, "iters": max_iter, "history": hist}


def fit_svm_primal(X, y, C = 1000, loss = "squared_hinge",
                   optimizer = "gd_armijo", step = 1e-2, alpha0 = 1,
                   tol = 1e-6, max_iter = 10000):
    """
    Train a primal SVM with a chosen first-order optimizer (fixed GD or Armijo GD)

    X: features, shape (n, d) without intercept
    y: labels in {0,1} or {-1,+1}  will be coerced to {-1,+1}
    C: regularization weight on the loss term
    loss: squared_hinge (smooth) or hinge (non-smooth)
    optimizer: "gd" with fixed step, "gd_armijo" with backtracking, "subgrad" for hinge
    step: fixed step size for "gd"
    alpha0: initial step guess for Armijo and base step for subgrad
    tol: stop when ||grad|| <= tol
    max_iter: iteration limit
    Returns: (theta, info) where theta = [w, b]
    """
        
    # ensure labels are {-1, +1}
    y = np.asarray(y)
    if set(np.unique(y)) == {0, 1}:
        y = 2*y - 1

    # add intercept to X and standardize types and shapes
    Xb = add_intercept(np.asarray(X, dtype=float))
    y = y.astype(float).reshape(-1)

    # pack the SVM objective
    # fg(theta) will return (loss_value, gradient_vector)
    fg = lambda theta: svm_loss_and_grad(theta, Xb, y, C = C, loss = loss)

    # initialize parameters to zeros and last entry is bias b
    theta0 = np.zeros(Xb.shape[1], dtype=float)

    # choose and run the selected optimizer
    if optimizer == "gd":
        # smooth objective with fixed GD
        theta, info = fit_gd_fixed(fg, theta0, step=step, tol=tol, max_iter=max_iter)
    elif optimizer == "gd_armijo":
        # smooth objective with Armijo backtracking
        if loss != "squared_hinge":
            raise ValueError("Armijo requires 'squared_hinge' (smooth).")
        theta, info = fit_gd_armijo(fg, theta0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    elif optimizer == "subgrad":
        # for non-smooth hinge
        theta, info = fit_subgradient(fg, theta0, alpha0=alpha0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError("optimizer must be {'gd','gd_armijo','subgrad'}")
    return theta, info

# Computes the raw decision function (f(x) = w^T x + b -> -1 or +1)
def svm_predict_score(theta, X):
    Xb = add_intercept(X)
    return Xb @ theta

# 1 if f(x) >= 0, -1 otherwise
def svm_predict_label(theta, X):
    return np.where(svm_predict_score(theta, X) >= 0, 1, -1)

# Compute mean accuracy with predicted y
def svm_accuracy(theta, X, y):
    y_hat = svm_predict_label(theta, X)
    if set(np.unique(y)) == {0,1}:   # match input coding
        y_hat = ((y_hat + 1) // 2).astype(int)
    return (y_hat == y).mean()

# Converts y label to -1, +1
def num_support_vectors(theta, X, y, eps=1e-6):
    y_pm = y.copy()
    if set(np.unique(y_pm)) == {0, 1}:
        y_pm = 2*y_pm - 1
    margins = y_pm * svm_predict_score(theta, X)
    return int(np.sum(margins <= 1.0 + eps))