import torch
import numpy as np
import utils.coord_systems as coords

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

def calculate_dop(position: np.ndarray | torch.Tensor, H: np.ndarray | torch.Tensor, W: np.ndarray | torch.Tensor) -> dict:
    """
    Calculate Dilution of Precision (DOP) from geometry matrix and weights.

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Receiver ECEF position (x, y, z).
    H : np.ndarray, shape (n_sats, 4)
        Geometry (Jacobian) matrix.
    W : np.ndarray, shape (n_sats, n_sats)
        Weight matrix (diagonal).

    Returns
    -------
    dop : dict
        Dictionary with keys: 'pdop', 'tdop', 'gdop', 'hdop', 'vdop'
    """
    n_sats = H.shape[0]
    if n_sats < 4:
        print("Warning: Not enough satellites for DOP calculation")
        return {'pdop': 999.9, 'tdop': 999.9, 'gdop': 999.9, 'hdop': 999.9, 'vdop': 999.9}

    # Covariance matrix: (H^T W H)^(-1)
    cov_matrix = np.linalg.inv(H.T @ W @ H)

    # Position and time variances
    sigma_x2, sigma_y2, sigma_z2, sigma_t2 = cov_matrix[0,0], cov_matrix[1,1], cov_matrix[2,2], cov_matrix[3,3]

    # PDOP, TDOP, GDOP
    pdop = np.sqrt(sigma_x2 + sigma_y2 + sigma_z2)
    tdop = np.sqrt(sigma_t2)
    gdop = np.sqrt(sigma_x2 + sigma_y2 + sigma_z2 + sigma_t2)

    # HDOP and VDOP (local ENU)
    ecef_to_enu = coords.ecef_to_enu_rot(position)   # 3x3 rotation matrix
    pos_cov = cov_matrix[0:3, 0:3]
    enu_cov = ecef_to_enu @ pos_cov @ ecef_to_enu.T

    hdop = np.sqrt(enu_cov[0,0] + enu_cov[1,1])
    vdop = np.sqrt(enu_cov[2,2])

    dop = {
        'pdop': pdop,
        'tdop': tdop,
        'gdop': gdop,
        'hdop': hdop,
        'vdop': vdop
    }

    return dop

def position(
    pr: np.ndarray,
    prr: np.ndarray,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    prev_estimate: dict | None = None,
    Wx: np.ndarray | None = None,
    Wv: np.ndarray | None = None
):
    """
    Estimate GNSS position and velocity using weighted least squares.

    Parameters
    ----------
    pr : pseudoranges, shape (n_sats,)
    prr : pseudorange rates, shape (n_sats,)
    sat_pos : satellite positions, shape (n_sats, 3)
    sat_vel : satellite velocities, shape (n_sats, 3)
    pr_uc : pseudorange uncertainties, shape (n_sats,)
    prr_uc : pseudorange rate uncertainties, shape (n_sats,)
    cn0_dbhz : carrier-to-noise ratios in dB-Hz, shape (n_sats,)
    prev_estimate : dict with keys 'position', 'velocity', 'clock_bias', 'clock_drift'
    Wx : weight matrix for position, shape (n_sats, n_sats)
    Wv : weight matrix for velocity, shape (n_sats, n_sats)

    Returns
    -------
    new_pos : dict with keys 'position', 'velocity', 'clock_bias', 'clock_drift', 'dop'
    """

    # Previous estimate fallback
    if prev_estimate is None:
        prev_estimate = {
            "position": np.array([0, 0, 6371000], dtype=float),
            "velocity": np.array([0, 0, 0], dtype=float),
            "clock_bias": 0.0,
            "clock_drift": 0.0,
        }

    if pr.size < 4:
        print("Warning: Not enough satellites for least squares. Returning previous position")
        return prev_estimate

    # Weight matrices
    if Wx is None:
        Wx = np.diag(np.ones(pr.shape[0]))
    if Wv is None:
        Wv = np.diag(np.ones(pr.shape[0]))

    # Initial guess
    x0 = np.zeros(4)
    x0[:3] = prev_estimate["position"]
    x0[3] = prev_estimate["clock_bias"]

    # Solve position
    pos_output = least_squares(
        x0,
        lambda x: pr_residuals(x, sat_pos, pr),
        lambda x: jacobian_residuals(x, sat_pos),
        Wx,
    )

    if np.isnan(pos_output).any():
        print("Warning: Least squares returned NaN for position. Returning previous estimate")
        return prev_estimate

    # Compute DOP
    H_pos = jacobian_residuals(pos_output, sat_pos)
    dop = calculate_dop(pos_output[:3], H_pos, Wx)
    if dop["pdop"] > 10.0:
        print(f"Rejecting poor quality solution: PDOP {dop['pdop']:.1f}")
        return prev_estimate

    # Initial guess for velocity
    v0 = np.zeros(4)
    v0[:3] = prev_estimate["velocity"]
    v0[3] = prev_estimate["clock_drift"]

    # Solve velocity
    vel_output = least_squares(
        v0,
        lambda v: prr_residuals(v, sat_vel, prr, pos_output, sat_pos),
        lambda _: jacobian_residuals(pos_output, sat_pos),
        Wv,
    )

    if np.isnan(vel_output).any():
        print("Warning: Least squares returned NaN for velocity. Returning previous estimate")
        return prev_estimate

    # Assemble output
    new_pos = {
        "position": pos_output[:3],
        "clock_bias": pos_output[3],
        "velocity": vel_output[:3],
        "clock_drift": vel_output[3],
        "dop": dop,
    }

    return new_pos

def position_torch(
    pr: torch.Tensor,
    prr: torch.Tensor,
    sat_pos: torch.Tensor,
    sat_vel: torch.Tensor,
    prev_estimate: dict | None = None,
    Wx: torch.Tensor | None = None,
    Wv: torch.Tensor | None = None,
):
    """
    Estimate GNSS position and velocity using weighted least squares (torch version).

    Parameters
    ----------
    pr : torch.Tensor, shape (n_sats,)
    prr : torch.Tensor, shape (n_sats,)
    sat_pos : torch.Tensor, shape (n_sats, 3)
    sat_vel : torch.Tensor, shape (n_sats, 3)
    prev_estimate : dict with keys 'position', 'velocity', 'clock_bias', 'clock_drift'
    Wx : torch.Tensor, weight matrix for position, shape (n_sats, n_sats)
    Wv : torch.Tensor, weight matrix for velocity, shape (n_sats, n_sats)

    Returns
    -------
    new_pos : dict with torch.Tensor entries
    """

    # Previous estimate fallback
    if prev_estimate is None:
        prev_estimate = {
            "position": torch.tensor([0.0, 0.0, 6371000.0], dtype=pr.dtype, device=pr.device),
            "velocity": torch.tensor([0.0, 0.0, 0.0], dtype=pr.dtype, device=pr.device),
            "clock_bias": torch.tensor(0.0, dtype=pr.dtype, device=pr.device),
            "clock_drift": torch.tensor(0.0, dtype=pr.dtype, device=pr.device),
        }

    if pr.numel() < 4:
        print("Warning: Not enough satellites for least squares. Returning previous position")
        return prev_estimate

    # Weight matrices
    if Wx is None:
        Wx = torch.eye(pr.shape[0], dtype=pr.dtype, device=pr.device)
    if Wv is None:
        Wv = torch.eye(prr.shape[0], dtype=pr.dtype, device=pr.device)

    # Initial guess
    x0 = torch.zeros(4, dtype=pr.dtype, device=pr.device)
    x0[:3] = prev_estimate["position"]
    x0[3] = prev_estimate["clock_bias"]

    # Solve position
    pos_output = least_squares_torch(
        x0,
        lambda x: pr_residuals_torch(x, sat_pos, pr),
        lambda x: jacobian_residuals_torch(x, sat_pos),
        Wx,
    )

    if torch.isnan(pos_output).any():
        print("Warning: Least squares returned NaN for position. Returning previous estimate")
        return prev_estimate

    # # Compute DOP (can be numpy or torch, depending on your implementation)
    # H_pos = jacobian_residuals_torch(pos_output, sat_pos)
    # dop = calculate_dop(pos_output[:3], H_pos, Wx)
    # if dop["pdop"] > 10.0:
    #     print(f"Rejecting poor quality solution: PDOP {dop['pdop']:.1f}")
    #     return prev_estimate

    # Initial guess for velocity
    v0 = torch.zeros(4, dtype=pr.dtype, device=pr.device)
    v0[:3] = prev_estimate["velocity"]
    v0[3] = prev_estimate["clock_drift"]

    # Solve velocity
    vel_output = least_squares_torch(
        v0,
        lambda v: prr_residuals_torch(v, sat_vel, prr, pos_output, sat_pos),
        lambda _: jacobian_residuals_torch(pos_output, sat_pos),
        Wv,
    )

    if torch.isnan(vel_output).any():
        print("Warning: Least squares returned NaN for velocity. Returning previous estimate")
        return prev_estimate

    # Assemble output
    new_pos = {
        "position": pos_output[:3],
        "clock_bias": pos_output[3],
        "velocity": vel_output[:3],
        "clock_drift": vel_output[3],
        "dop": {'pdop': 999.9, 'tdop': 999.9, 'gdop': 999.9, 'hdop': 999.9, 'vdop': 999.9},
    }

    return new_pos
