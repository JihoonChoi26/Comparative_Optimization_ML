### Optimizer Setup for Logistic Regression with L2 Penalty
import numpy as np

# Armijo Backtracking (Approximated Line Search)
def armijo_backtracking(fg, x, p, f0=None, g0=None, alpha0=1.0, c1=1e-4, tau=0.5, max_bt=20):
    """
    fg: function that returns (f(x), grad f(x))
    x: current point (vector)
    p: search direction (typically -grad)
    f0, g0: optional cached f(x), grad f(x) at the current x to avoid recompute
    alpha0: initial step size guess
    c1: Armijo constant in (0, 1). Smaller means stricter decrease requirement
    tau: backtracking shrink factor in (0, 1). Default is 0.5
    max_bt: maximum backtracking iterations
    returns alpha or None if no satisfactory step found
    """

    # use predefined f and g if given, if not, compute those
    if f0 is None or g0 is None:
        f0, g0 = fg(x)
    
    # directional derivative at x along p
    gTp = float(np.dot(g0, p))
    
    # start step size
    alpha = float(alpha0)

    # Armijo condition: f(x _ alpha * p) <= f(x) + c1 * alpha * g^T * p
    for _ in range(max_bt):
        x_try = x + alpha * p
        f_try, _ = fg(x_try)
        if f_try <= f0 + c1 * alpha * gTp:
            return alpha # sufficient decrease achieved
        alpha *= tau # if not, shrink and try again
    
    return None  # give up if could not satisfy sufficient decrease with max_bt

# Gradient Descent with fixed step size
def fit_gd_fixed(fg, w0, step=1e-2, tol=1e-6, max_iter=10000):
    """
    fg: function that returns (loss, grad) at w
    w0: starting weights
    step: fixed step size
    tol: stop when ||grad||_2 <= tol
    max_iter: cap on iterations
    returns (w, info) where info has status, iters, history[(k, loss, ||grad||)]
    """
    w = w0.copy()
    hist = []

    for k in range(max_iter):
        f, g = fg(w) # current loss and gradient
        gnorm = float(np.linalg.norm(g)) # gradient norm for stopping
        hist.append((k, f, gnorm))

        # convergence check
        if np.linalg.norm(g) <= tol:
            return w, {"status": 0, "iters": k, "history": hist}
        
        w = w - step * g # fixed step update
    
    # did not converged within max_iter    
    return w, {"status": 1, "iters": max_iter, "history": hist}

# Gradient Descent with Armijo Backtracking
def fit_gd_armijo(fg, w0, alpha0=1.0, tol=1e-6, max_iter=10_000,
                  c1=1e-4, tau=0.5, max_bt=20):    
    """
    fg: function that returns (loss, grad) at w
    w0: starting weights
    alpha0: initial step guess each iteration
    tol: stop when ||grad||_2 <= tol
    max_iter: max iterations
    c1, tau, max_bt: Armijo parameters

    returns (w, info) where info has status, iters, history[(k, loss, ||grad||)]
    """ 
    # working copy
    w = w0.copy()
    
    # history for plotting
    hist = []

    for k in range(max_iter):
        # evaluate objective and gradient at current point
        f, g = fg(w)

        # log progress
        hist.append((k, f, np.linalg.norm(g)))

        # first-order stopping rule
        if np.linalg.norm(g) <= tol:
            return w, {"status": 0, "iters": k, "history": hist}

        # choose the steepest descent direction
        p = -g

        # pic a step length using Armijo backtracking
        alpha = armijo_backtracking(fg, w, p, f0 = f, g0 = g, alpha0 = alpha0, 
                                    c1 = c1, tau = tau, max_bt = max_bt)
        # take a tiny step if line search failed
        if alpha is None:
            alpha = 1e-6

        # update iterate
        w = w + alpha * p
    return w, {"status": 1, "iters": max_iter, "history": hist}