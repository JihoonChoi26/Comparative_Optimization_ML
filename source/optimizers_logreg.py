### Optimizer Setup for Logistic Regression with L2 Penalty
import numpy as np

# Gradient Descent with fixed step size
def fit_gd_fixed(fg, w0, step=1e-2, tol=1e-6, max_iter=10000):
    """
    Gradient Descent with a fixed step size.
    
    Inputs
        fg: function that returns (loss, grad) at w
        w0: starting weights
        step: fixed step size
        tol: stop when ||grad||_2 <= tol
        max_iter: cap on iterations
    Returns
        (w, info) where info has status, iters, history[(k, loss, ||grad||)]
    """
    w = w0.copy()
    hist = []
    
    for k in range(max_iter):
        # Compute function value and get loss and gradient at current point
        f, g = fg(w)
        gnorm = float(np.linalg.norm(g)) # Calculate gradient norm
        hist.append((k, f, gnorm)) # Save iteration info
            
        # Stop if gradient is small enough
        if np.linalg.norm(g) <= tol:
            return w, {'status': 0, 'iters': k, 'history': hist}
        
        # Update current point with fixed step
        w = w - step * g 
            
    # Return if max iterations reached without convergence
    return w, {'status': 1, 'iters': max_iter, 'history': hist}

# Armijo Backtracking (Approximated Line Search)
def armijo_backtracking(fg, x, p, f0=None, g0=None, alpha0=1.0, c1=1e-4, tau=0.5, max_bt=20):
    """
    Performs Armijo backtracking line search to find a good step size.
    Ensures that the objective function decreases sufficiently.
    
    Inputs
        fg: function that returns (f(x), grad f(x))
        x: current point (vector)
        p: search direction (typically -grad)
        f0, g0: cached f(x), grad f(x) at the current x to avoid recompute
        alpha0: initial step size guess
        c1: Armijo constant in (0, 1). Smaller means stricter decrease requirement
        tau: backtracking shrink factor in (0, 1). Default is 0.5
        max_bt: maximum backtracking iterations
    
    Returns
        alpha or None if no satisfactory step found
    """
    
    # Compute function value and gradien at x if not given
    if f0 is None or g0 is None:
        f0, g0 = fg(x)
        
    # Directional derivative at x along p
    gTp = float(np.dot(g0, p))
    
    # Start with initial step size
    alpha = float(alpha0)
        
    # Backtracking loop to check Armijo condition for sufficient decrease
    # Armijo condition: f(x _ alpha * p) <= f(x) + c1 * alpha * g^T * p
    for _ in range(max_bt):
        x_try = x + alpha * p # Candidate point
        f_try, _ = fg(x_try) # Evaluate function at candidate
        # If Armijo condition satisfied
        if f_try <= f0 + c1 * alpha * gTp:
            return alpha # Sufficient decrease achieved; step size found
        alpha *= tau # if not, shrink and try again
        
    # Return None if no good step found
    return None

# Gradient Descent with Armijo Backtracking
def fit_gd_armijo(fg, w0, alpha0=1.0, tol=1e-6, max_iter=10_000, c1=1e-4, tau=0.5, max_bt=20):    
    """
    Gradient Descent with adaptive step size (Armijo).
    
    Inputs
        fg: function that returns (loss, grad) at w
        w0: starting weights
        alpha0: initial step guess each iteration
        tol: stop when ||grad||_2 <= tol
        max_iter: max iterations
        c1, tau, max_bt: Armijo parameters

    Returns
        (w, info) where info has status, iters, history[(k, loss, ||grad||)]
    """ 
    w = w0.copy() # Make a copy of initial point
    hist = [] # Store history for plotting
        
    for k in range(max_iter):
        # Compute function value and get loss and gradient at current point
        f, g = fg(w)
        hist.append((k, f, np.linalg.norm(g))) # Save iteration info
            
        # Stop if gradient is small enough
        if np.linalg.norm(g) <= tol:
            return w, {'status': 0, 'iters': k, 'history': hist}
        
        # Use steepest descent direction
        p = -g
            
        # Find step size using Armijo backtracking
        alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=alpha0, c1=c1, tau=tau, max_bt=max_bt)
        # If backtracking fails, use tiny step
        if alpha is None:
            alpha = 1e-6
            
        # Update current point
        w = w + alpha * p
        
    # Return if max iterations reached without convergence
    return w, {'status': 1, 'iters': max_iter, 'history': hist}

# Nonlinear Conjugate Gradient with Polak-Ribiere Plus (PR+)
def fit_nlcg_prp(fg, w0, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    Nonlinear Conjugate Gradient using Polak-Ribiere Plus (PR+) update.
    This method uses previous search directions to accelerate convergence.
    
    Inputs
        fg: function; returns (loss, gradient)
        w0: array; initial weights
        tol: float; stopping tolerance for gradient norm
        max_iter: int; maximum iterations
        alpha0: float; initial step size guess for line search
    
    Returns
        (w, info) where info has status, iters, history
    """
    w = w0.copy() # Make a copy of initial point
    f, g = fg(w)
    
    # Initial direction
    p = -g # steepest descent
    
    hist = [] # Store history for plotting
    hist.append((0, f, np.linalg.norm(g)))
    
    # Store previous gradient for PR+ calculation
    g_old = g.copy()
    
    # To handle restart strategy (for robustness)
    n_dim = len(w)
        
    for k in range(1, max_iter + 1):
        gnorm = np.linalg.norm(g)
        
        # 1) Check convergence
        if gnorm <= tol:
            return w, {'status': 0, 'iters': k, 'history': hist}
        
        # 2) Line Search (reuse Armijo from previous implementation)
        # Using Armijo; although CG often prefers strong Wolfe, Armijo is often sufficient for simple logistic regression
        alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=alpha0)
        
        if alpha is None:
            # Line search failed to find a step -> stop
            print(f'NLCG line search failed at iter {k}')
            return w, {'status': 2, 'iters': k, 'history': hist}
        
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
        
    return w, {'status': 1, 'iters': max_iter, 'history': hist}

# Quasi-Newton Method (BFGS)
def fit_bfgs(fg, w0, tol=1e-6, max_iter=10000, alpha0=1.0):
    """
    BFGS (Quasi-Newton) Method.
    Approximates the inverse Hessian matrix iteratively for faster convergence.
    
    Inputs
        fg: function; returns (loss, gradient)
        w0: array; initial weights
        tol: float; stopping tolerance
        max_iter: int; maximum iterations
        alpha0: float; initial step size; for Newton-type methods, 1.0 is the ideal starting guess

    Returns
        (w, info) where info has status, iters, history
    """
    w = w0.copy()
    n_dim = len(w)
    
    # Initial Hessian approximation
    H = np.eye(n_dim, dtype=float) # Inverse Hessian H_0 = I
    
    f, g = fg(w)
    hist = []
    hist.append((0, f, np.linalg.norm(g)))
    
    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) <= tol:
            return w, {'status': 0, 'iters': k, 'history': hist}
        
        # 1) Determine Search Direction: p_k = - H_k * g_k
        p = -H @ g
        
        # 2) Line Search
        # Quasi-Newton; trying alpha=1.0 first (Quadratic convergence property)
        alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=1.0)
        
        if alpha is None:
            # Fallback
            # If step 1.0 fails significantly, try smaller
            alpha = armijo_backtracking(fg, w, p, f0=f, g0=g, alpha0=1e-2)
            if alpha is None:
                print(f'BFGS line search failed at iter {k}')
                return w, {'status': 2, 'iters': k, 'history': hist}
        
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
        
    return w, {'status': 1, 'iters': max_iter, 'history': hist}