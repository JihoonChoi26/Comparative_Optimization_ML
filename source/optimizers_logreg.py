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

# Nonlinear Conjugate Gradient with Polak-Ribiere Plus (PR+)
def fit_nlcg_prp(fg, w0, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    Minimizes f(w) using Nonlinear Conjugate Gradient (Polak-Ribiere+).
    
    Parameters
    ----------
    fg : function
        Returns (loss, gradient).
    w0 : array
        Initial weights.
    tol : float
        Stopping tolerance for gradient norm.
    max_iter : int
        Maximum iterations.
    alpha0 : float
        Initial step size guess for line search.
    
    Returns
    -------
    w : array
        Final weights.
    info : dict
        Optimization info containing history and status.
    """
    w = w0.copy()
    f, g = fg(w)
    
    # Initial direction
    # steepest descent
    p = -g 
    
    hist = []
    hist.append((0, f, np.linalg.norm(g)))
    
    # Store previous gradient for PR+ calculation
    g_old = g.copy()
    
    # To handle restart strategy (for robustness)
    n_dim = len(w)
    
    for k in range(1, max_iter + 1):
        gnorm = np.linalg.norm(g)
        
        # 1) Check convergence
        if gnorm <= tol:
            return w, {"status": 0, "iters": k, "history": hist}
        
        # 2) Line Search (reuse Armijo from previous implementation)
        # (Although CG often prefers strong Wolfe, Armijo is often sufficient for simple logistic regression)
        alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=alpha0)
        
        if alpha is None:
            # Line search failed to find a step -> stop
            print(f"NLCG line search failed at iter {k}")
            return w, {"status": 2, "iters": k, "history": hist}
        
        # 3) Update position
        w_new = w + alpha * p
        f_new, g_new = fg(w_new)
        
        hist.append((k, f_new, np.linalg.norm(g_new)))
        
        # 4) Compute Beta (Polak-Ribiere Plus)
        # diff = g_{k+1} - g_k
        g_diff = g_new - g_old
        
        # Denominator: g_k^T g_k
        denom = np.dot(g_old, g_old)
        
        if denom == 0:
            beta = 0.0
        else:
            # Numerator: g_{k+1}^T (g_{k+1} - g_k)
            numer = np.dot(g_new, g_diff)
            beta_pr = numer / denom
            # PR+ condition: beta = max(0, beta_pr)
            beta = max(0.0, beta_pr)
        
        # 5) Update direction
        # p_{k+1} = -g_{k+1} + beta * p_k
        
        # Restart every n iterations or when direction isn't looking good
        # Briefly check the descent condition or trust PR+ and restart
        p_new = -g_new + beta * p
        
        # Check if p_new is a descent direction (g^T p < 0)
        # If not, reset to -g
        if np.dot(g_new, p_new) >= 0:
             p_new = -g_new
        
        # Update references for next iteration
        w = w_new
        f = f_new
        g_old = g_new.copy() # Store current g as old for next step
        g = g_new # Update current g
        p = p_new
        
    return w, {"status": 1, "iters": max_iter, "history": hist}

# Quasi-Newton Method (BFGS)
def fit_bfgs(fg, w0, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    Minimizes f(w) using BFGS (Quasi-Newton).
    
    Parameters
    ----------
    fg : function
        Returns (loss, gradient).
    w0 : array
        Initial weights.
    tol : float
        Stopping tolerance.
    max_iter : int
        Maximum iterations.
    alpha0 : float
        Initial step size. For Newton-type methods, 1.0 is the ideal starting guess.

    Returns
    -------
    w : array
        Final weights.
    info : dict
        Optimization info.
    """
    w = w0.copy()
    n_dim = len(w)
    
    # Initial Hessian approximation
    # (Inverse Hessian H_0 = I)
    H = np.eye(n_dim, dtype=float)
    
    f, g = fg(w)
    hist = []
    hist.append((0, f, np.linalg.norm(g)))
    
    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) <= tol:
            return w, {"status": 0, "iters": k, "history": hist}
        
        # 1) Determine Search Direction: p_k = - H_k * g_k
        p = -H @ g
        
        # 2) Line Search
        # Quasi-Newton; trying alpha=1.0 first (Quadratic convergence property)
        alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=1.0)
        
        if alpha is None:
            # Fallback
            # if step 1.0 fails significantly, try smaller
            alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=1e-2)
            if alpha is None:
                print(f"BFGS line search failed at iter {k}")
                return w, {"status": 2, "iters": k, "history": hist}
        
        # 3) Update parameters
        # s_k = w_{k+1} - w_k
        s = alpha * p
        w_new = w + s
        
        f_new, g_new = fg(w_new)
        hist.append((k, f_new, np.linalg.norm(g_new)))
        
        # 4) Update Inverse Hessian (H) using BFGS
        # y_k = g_{k+1} - g_k
        y = g_new - g
        
        # Curvature condition: y^T s > 0
        yTs = np.dot(y, s)
        
        if yTs > 1e-10: # Ensure strictly positive to maintain positive definiteness
            rho = 1.0 / yTs
            I = np.eye(n_dim)
            
            # BFGS formula: H_{k+1} = (I - rho * s * y^T) H_k (I - rho * y * s^T) + rho * s * s^T
            
            # Term A = I - rho * s * y^T
            # Outer product s*y^T is (d,1) @ (1,d) -> (d,d)
            A = I - rho * np.outer(s, y)
            
            # Term B = I - rho * y * s^T
            B = I - rho * np.outer(y, s)
            
            # H_new = A @ H @ B + rho * s * s^T
            H = A @ H @ B + rho * np.outer(s, s)
            
        else:
            # If curvature condition is not met, skip update or reset H
            pass
            
        # Update current state
        w = w_new
        f = f_new
        g = g_new
        
    return w, {"status": 1, "iters": max_iter, "history": hist}