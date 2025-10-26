import torch
import numpy as np

# Constants
EARTH_ROTATION_SPEED = 7.292115e-5  # rad/s
SPEED_OF_LIGHT = 299792458.0        # m/s

def los_vector(xusr: np.ndarray, xsat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute line-of-sight unit vectors and ranges.
    
    Parameters
    ----------
    xusr : ndarray, shape (1, 3) or (3,)
        User position
    xsat : ndarray, shape (n_sat, 3)
        Satellite positions
    
    Returns
    -------
    u : ndarray, shape (n_sat, 3)
        Unit LOS vectors
    rng : ndarray, shape (n_sat,)
        Ranges
    """
    u = xsat - xusr.reshape(1, 3)
    rng = np.linalg.norm(u, axis=1)
    u /= rng[:, np.newaxis]  # row-wise normalization
    return u, rng


def jacobian_residuals(x: np.ndarray, xsat: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of pseudorange residuals.
    
    Parameters
    ----------
    x : ndarray, shape (4,)
        State vector [x, y, z, clock_bias]
    xsat : ndarray, shape (n_sat, 3)
        Satellite positions
    
    Returns
    -------
    J : ndarray, shape (n_sat, 4)
        Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.zeros((xsat.shape[0], 4))
    J[:, :3] = -u
    J[:, 3] = 1.0
    return J


def pr_residuals(x: np.ndarray, xsat: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """
    Compute pseudorange residuals.
    
    Parameters
    ----------
    x : ndarray, shape (4,)
        State vector [x, y, z, clock_bias]
    xsat : ndarray, shape (n_sat, 3)
        Satellite positions
    pr : ndarray, shape (n_sat,)
        Measured pseudoranges
    
    Returns
    -------
    residuals : ndarray, shape (n_sat,)
    """
    u, rng = los_vector(x[:3], xsat)

    # Sagnac effect
    rng += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0])

    residuals = rng - (pr - x[3])
    return residuals


def prr_residuals(v: np.ndarray, vsat: np.ndarray, prr: np.ndarray, x: np.ndarray, xsat: np.ndarray) -> np.ndarray:
    """
    Compute pseudorange rate residuals.
    
    Parameters
    ----------
    v : ndarray, shape (4,)
        User velocity [vx, vy, vz, clock_rate]
    vsat : ndarray, shape (n_sat, 3)
        Satellite velocities
    prr : ndarray, shape (n_sat,)
        Measured pseudorange rates
    x : ndarray, shape (4,)
        User position [x, y, z, clock_bias]
    xsat : ndarray, shape (n_sat, 3)
        Satellite positions
    
    Returns
    -------
    residuals : ndarray, shape (n_sat,)
    """
    u, _ = los_vector(x[:3], xsat)
    rate = np.zeros(xsat.shape[0])

    for i in range(xsat.shape[0]):
        rate[i] = np.dot(vsat[i, :3] - v[:3], u[i])
        rate[i] += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
            vsat[i, 1] * x[0] + xsat[i, 1] * v[0] - vsat[i, 0] * x[1] - xsat[i, 0] * v[1]
        )

    residuals = rate - (prr - v[3])
    return residuals

def least_squares(v0: np.ndarray, residuals_func, jacobian_func, W: np.ndarray, max_iters: int=50, tol: float=1e-6):
    """
    Weighted least squares solver using iterative normal equations.

    Parameters
    ----------
    v0 : np.ndarray, shape (n,)
        Initial guess.
    residuals_func : callable
        Function returning residuals vector r(v), shape (m,).
    jacobian_func : callable
        Function returning Jacobian matrix H(v), shape (m, n).
    W : np.ndarray, shape (m, m)
        Weight matrix (assumed symmetric positive definite).
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on delta norm.

    Returns
    -------
    v : np.ndarray, shape (n,)
        Estimated solution vector.
    """
    v = v0.copy()

    for _ in range(max_iters):
        r = residuals_func(v)          # shape (m,)
        H = jacobian_func(v)           # shape (m, n)

        # Apply weighting: r_w = W r, H_w = W H
        r_w = W @ r                    # weighted residuals
        H_w = W @ H                    # weighted Jacobian

        # Solve normal equations: (H^T H) delta = -H^T r
        HTH = H_w.T @ H_w
        JTr = H_w.T @ r_w
        delta = -np.linalg.solve(HTH, JTr)

        v += delta

        if np.linalg.norm(delta) < tol:
            break

    return v

##########################################################################################
#
# The following are PyTorch versions of the above functions for use with pytorch autograd.
# They mirror the numpy implementations but use torch tensors and maths functions.
# Torch tensors shouldn't be used normally since they are slower to work with.
#
##########################################################################################

def los_vector_torch(xusr: torch.Tensor, xsat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = xsat - xusr.view(1, 3)
    rng = torch.norm(u, dim=1)
    u = u / rng.view(-1, 1)
    return u, rng

def jacobian_residuals_torch(x: torch.Tensor, xsat: torch.Tensor) -> torch.Tensor:
    u, _ = los_vector_torch(x[:3], xsat)
    J = torch.zeros((xsat.shape[0], 4), dtype=x.dtype, device=x.device)
    J[:, :3] = -u
    J[:, 3] = 1.0
    return J

def pr_residuals_torch(x: torch.Tensor, xsat: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    u, rng = los_vector_torch(x[:3], xsat)
    rng = rng + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0])
    residuals = rng - (pr - x[3])
    return residuals

def prr_residuals_torch(v: torch.Tensor, vsat: torch.Tensor, prr: torch.Tensor, x: torch.Tensor, xsat: torch.Tensor) -> torch.Tensor:
    u, _ = los_vector_torch(x[:3], xsat)
    rate = torch.zeros(xsat.shape[0], dtype=x.dtype, device=x.device)

    for i in range(xsat.shape[0]):
        rate[i] = torch.dot(vsat[i, :3] - v[:3], u[i])
        rate[i] += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
            vsat[i, 1] * x[0] + xsat[i, 1] * v[0] - vsat[i, 0] * x[1] - xsat[i, 0] * v[1]
        )

    residuals = rate - (prr - v[3])
    return residuals

def least_squares_torch(v0: torch.Tensor, residuals_func, jacobian_func, W: torch.Tensor, max_iters: int=50, tol: float=1e-6):
    v = v0.clone()

    for _ in range(max_iters):
        r = residuals_func(v)           # shape (m,)
        H = jacobian_func(v)            # shape (m, n)

        # Apply weighting
        r_w = W @ r                     # weighted residual
        H_w = W @ H                     # weighted Jacobian

        # Solve normal equations: (H^T H) delta = -H^T r
        HTH = H_w.T @ H_w               # (n, n)
        HTr = H_w.T @ r_w               # (n,)
        delta = -torch.linalg.solve(HTH, HTr)

        v = v + delta

        if delta.norm() < tol:
            break

    return v