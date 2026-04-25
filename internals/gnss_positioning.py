import torch
import numpy as np
from internals.constants import EARTH_ROTATION_SPEED, SPEED_OF_LIGHT
import internals.coord_systems as coords
from numba import njit

@njit(fastmath=True)
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
    Tuple[ndarray, ndarray]
        A tuple containing the unit LOS vectors and ranges.
    """
    n = xsat.shape[0]
    u = np.empty((n, 3), dtype=np.float64)
    rng = np.empty(n, dtype=np.float64)

    for i in range(n):
        dx = xsat[i, 0] - xusr[0]
        dy = xsat[i, 1] - xusr[1]
        dz = xsat[i, 2] - xusr[2]

        r = (dx * dx + dy * dy + dz * dz) ** 0.5
        rng[i] = r

        inv_r = 1.0 / r
        u[i, 0] = dx * inv_r
        u[i, 1] = dy * inv_r
        u[i, 2] = dz * inv_r

    return u, rng


@njit(fastmath=True)
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
    ndarray, shape (n_sat, 4)
        Jacobian matrix of residuals with respect to state.
    """
    u, _ = los_vector(x[:3], xsat)
    n = xsat.shape[0]

    J = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        J[i, 0] = -u[i, 0] + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * xsat[i, 1]
        J[i, 1] = -u[i, 1] - EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * xsat[i, 0]
        J[i, 2] = -u[i, 2]
        J[i, 3] = 1.0

    return J


@njit(fastmath=True)
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
    ndarray, shape (n_sat,)
        Pseudorange residuals.
    """
    _, rng = los_vector(x[:3], xsat)
    n = xsat.shape[0]

    sagnac = EARTH_ROTATION_SPEED / SPEED_OF_LIGHT
    residuals = np.empty(n, dtype=np.float64)

    for i in range(n):
        rng_i = rng[i] + sagnac * (
            xsat[i, 0] * x[1] - xsat[i, 1] * x[0]
        )
        residuals[i] = rng_i - (pr[i] - x[3])

    return residuals


@njit(fastmath=True)
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
    ndarray, shape (n_sat,)
        Pseudorange rate residuals.
    """
    u, _ = los_vector(x[:3], xsat)
    n = xsat.shape[0]

    residuals = np.empty(n, dtype=np.float64)
    sagnac = EARTH_ROTATION_SPEED / SPEED_OF_LIGHT

    for i in range(n):
        rate = (
            (vsat[i, 0] - v[0]) * u[i, 0]
            + (vsat[i, 1] - v[1]) * u[i, 1]
            + (vsat[i, 2] - v[2]) * u[i, 2]
        )

        rate += sagnac * (
            vsat[i, 1] * x[0] + xsat[i, 1] * v[0]
            - vsat[i, 0] * x[1] - xsat[i, 0] * v[1]
        )

        residuals[i] = rate - (prr[i] - v[3])

    return residuals

def least_squares(
    v0: np.ndarray, 
    residuals_func: callable, 
    jacobian_func: callable, 
    W: np.ndarray, 
    huber_delta: float, 
    max_iters: int=50, 
    tol: float=1e-6
) -> np.ndarray:
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
    ndarray, shape (n,)
        Estimated state vector minimizing weighted residuals.
    """
    v = v0.copy()

    for _ in range(max_iters):
        r = residuals_func(v)      # (m,)
        H = jacobian_func(v)       # (m, n)

        abs_r = np.abs(r)
        huber_w = np.where(abs_r <= huber_delta, 1.0, huber_delta / (abs_r + 1e-8))
        W_huber_diag = np.diagonal(W) * huber_w
        W_huber = np.diag(W_huber_diag)

        # Apply weighting
        r_w = W_huber @ r                # weighted residual
        H_w = W_huber @ H                # weighted Jacobian

        # Solve normal equations: (H^T H) delta = -H^T r
        delta, _, _, _ = np.linalg.lstsq(H_w, -r_w, rcond=None)

        v_new = v + delta

        # Check convergence
        if np.linalg.norm(delta) < tol:
            v = v_new
            break

        v = v_new

    return v


def calculate_dop(position: np.ndarray, H: np.ndarray, W: np.ndarray) -> dict:
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
    dict
        A dictionary containing PDOP, TDOP, GDOP, HDOP, and VDOP values.
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
    ecef_to_enu = coords.ecef_to_enu_rot(position) # 3x3 rotation matrix
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
    prev_estimate : dict with keys 'position', 'velocity', 'clock_bias', 'clock_drift'
    Wx : weight matrix for position, shape (n_sats, n_sats)
    Wv : weight matrix for velocity, shape (n_sats, n_sats)

    Returns
    -------
    Dict
        A dictionary containing the new position, velocity, clock bias, clock drift, and DOP values.
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
        huber_delta=50.0
    )

    if np.isnan(pos_output).any():
        print("Warning: Least squares returned NaN for position. Returning previous estimate")
        return prev_estimate

    # Compute DOP
    H_pos = jacobian_residuals(pos_output, sat_pos)
    dop = calculate_dop(pos_output[:3], H_pos, Wx)
    if dop["pdop"] > 10.0:
        print(f"Warning: poor quality solution: PDOP {dop['pdop']:.1f}")
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
        huber_delta=5.0
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

def estimate_clock_bias(
    x_true: np.ndarray,
    xsat: np.ndarray,
    pr: np.ndarray
) -> np.ndarray:
    """
    Estimate clock bias (c * delta_t) in metres via weighted pseudoinverse projection using truth geometry.
    
    Parameters
    ----------
    x_true: ndarray, shape (4,)
        True state vector [x, y, z, clock_bias].
    xsat: ndarray, shape (n_sat, 3)
        Satellite positions.
    pr: ndarray, shape (n_sat,)
        Measured pseudoranges.

    Returns
    -------
    ndarray
        Estimated clock bias in metres.
    """
    m = pr.shape[0]
    W = np.eye(m, dtype=pr.dtype)

    # Ensure we only use true position
    x_pos = x_true[:3]

    # Line-of-sight and geometric range
    _, rng = los_vector(x_pos, xsat)

    # Apply Earth rotation (Sagnac)
    rng = rng + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
        xsat[:, 0] * x_pos[1] - xsat[:, 1] * x_pos[0]
    )

    # Pseudorange error (measured - true geometry)
    delta_rho = pr - rng

    H = jacobian_residuals(x_true, xsat)

    H_w = W @ H
    HTH = H_w.T @ H_w

    # Weighted pseudoinverse
    H_pinv = np.linalg.solve(HTH, H_w.T @ W)  # (4, m)

    # Final row corresponds to clock
    p_delta_t = H_pinv[3]  # (m,)

    delta_clock_m = p_delta_t @ delta_rho

    return delta_clock_m

def estimate_clock_drift(
    x_true: np.ndarray,
    v_true: np.ndarray,
    xsat: np.ndarray,
    vsat: np.ndarray,
    prr: np.ndarray,
) -> np.ndarray:
    """
    Estimate clock drift (c * delta_t_dot) in metres per second via weighted pseudoinverse projection using truth geometry.

    Parameters
    ----------
    x_true: ndarray, shape (4,)
        True state vector [x, y, z, clock_bias].
    v_true: ndarray, shape (4,)
        True velocity vector [vx, vy, vz, clock_drift].
    xsat: ndarray, shape (n_sat, 3)
        Satellite positions.
    vsat: ndarray, shape (n_sat, 3)
        Satellite velocities.
    prr: ndarray, shape (n_sat,)
        Measured pseudorange rates.

    Returns
    -------
    ndarray
        Estimated clock drift in metres per second.
    """

    m = prr.shape[0]
    W = np.eye(m, dtype=prr.dtype)

    # True LOS vectors
    u, _ = los_vector(x_true[:3], xsat)

    # True geometric range rate
    rate = np.zeros(m, dtype=prr.dtype)

    for i in range(m):
        rate[i] = np.dot(vsat[i, :3] - v_true[:3], u[i])
        rate[i] += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
            vsat[i, 1] * x_true[0]
            + xsat[i, 1] * v_true[0]
            - vsat[i, 0] * x_true[1]
            - xsat[i, 0] * v_true[1]
        )

    # Pseudorange rate error (measured - true geometry)
    delta_rho_dot = prr - rate

    # PRR Jacobian wrt velocity state
    H = np.zeros((m, 4), dtype=prr.dtype)
    H[:, :3] = -u
    H[:, 3] = 1.0

    # Weighted pseudoinverse
    H_w = W @ H
    HTH = H_w.T @ H_w
    H_pinv = np.linalg.solve(HTH, H_w.T @ W)

    # Clock drift row
    p_delta_t_dot = H_pinv[3]

    delta_clock_drift_mps = p_delta_t_dot @ delta_rho_dot

    return delta_clock_drift_mps

##########################################################################################
#
# The following are PyTorch versions of the above functions for use with pytorch autograd.
# They mirror the other implementations but use torch tensors and maths functions.
# Torch tensors shouldn't be used normally since they are slower to work with.
#
##########################################################################################

def los_vector_torch(xusr: torch.Tensor, xsat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate line-of-sight vector and geometric range between user and satellites.

    Parameters
    ----------
    xusr : torch.Tensor, shape (3,)
        User position.
    xsat : torch.Tensor, shape (n_sat, 3)
        Satellite positions.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the unit LOS vectors and ranges.
    """
    u = xsat - xusr.view(1, 3)
    rng = torch.norm(u, dim=1)
    u = u / rng.view(-1, 1)
    return u, rng

def jacobian_residuals_torch(x: torch.Tensor, xsat: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian of pseudorange residuals

    Parameters
    ----------
    x : torch.Tensor, shape (4,)
        State vector [x, y, z, clock_bias]
    xsat : torch.Tensor, shape (n_sat, 3)
        Satellite positions

    Returns
    -------
    torch.Tensor, shape (n_sat, 4)
        Jacobian matrix of residuals with respect to state.
    """
    u, _ = los_vector_torch(x[:3], xsat)
    J = torch.zeros((xsat.shape[0], 4), dtype=x.dtype, device=x.device)
    J[:, 0] = -u[:, 0] - EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * xsat[:, 1]
    J[:, 1] = -u[:, 1] + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * xsat[:, 0]
    J[:, 2] = -u[:, 2]
    J[:, 3] = 1.0
    return J

def pr_residuals_torch(x: torch.Tensor, xsat: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    """
    Compute pseudorange residuals.

    Parameters
    ----------
    x : torch.Tensor, shape (4,)
        State vector [x, y, z, clock_bias]
    xsat : torch.Tensor, shape (n_sat, 3)
        Satellite positions
    pr : torch.Tensor, shape (n_sat,)
        Measured pseudoranges

    Returns
    -------
    torch.Tensor, shape (n_sat,)
        Pseudorange residuals.
    """
    _, rng = los_vector_torch(x[:3], xsat)
    rng = rng + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0])
    residuals = rng - (pr - x[3])
    return residuals

def prr_residuals_torch(v: torch.Tensor, vsat: torch.Tensor, prr: torch.Tensor, x: torch.Tensor, xsat: torch.Tensor) -> torch.Tensor:
    """
    Compute pseudorange rate residuals.

    Parameters
    ----------
    v : torch.Tensor, shape (4,)
        Velocity vector [vx, vy, vz, clock_drift]
    vsat : torch.Tensor, shape (n_sat, 3)
        Satellite velocities
    prr : torch.Tensor, shape (n_sat,)
        Measured pseudorange rates
    x : torch.Tensor, shape (4,)
        State vector [x, y, z, clock_bias]
    xsat : torch.Tensor, shape (n_sat, 3)
        Satellite positions

    Returns
    -------
    torch.Tensor, shape (n_sat,)
        Pseudorange rate residuals.
    """
    u, _ = los_vector_torch(x[:3], xsat)
    rate = torch.zeros(xsat.shape[0], dtype=x.dtype, device=x.device)

    for i in range(xsat.shape[0]):
        rel_vel = vsat[:, :3] - v[:3].unsqueeze(0)          # (m, 3)
        rate = (rel_vel * u).sum(dim=1)                       # (m,)
        rate += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
            vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
        - vsat[:, 0] * x[1] - xsat[:, 0] * v[1]
        )

    residuals = rate - (prr - v[3])
    return residuals

def least_squares_torch(
    v0: torch.Tensor, 
    residuals_func: callable, 
    jacobian_func: callable, 
    W: torch.Tensor, 
    huber_delta: float, 
    max_iters: int=50, 
    tol: float=1e-6
) -> torch.Tensor:
    """
    Weighted least squares solver using iterative normal equations.

    Parameters
    ----------
    v0 : torch.Tensor, shape (n,)
        Initial guess.
    residuals_func : callable
        Function returning residuals vector r(v), shape (m,).
    jacobian_func : callable
        Function returning Jacobian matrix H(v), shape (m, n).
    W : torch.Tensor, shape (m, m)
        Weight matrix (assumed symmetric positive definite).
    huber_delta : float
        Huber loss delta parameter for robust weighting.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on delta norm.

    Returns
    -------
    torch.Tensor, shape (n,)
        Estimated state vector minimizing weighted residuals.
    """
    v = v0.clone()

    for _ in range(max_iters):
        r = residuals_func(v)           # shape (m,)
        H = jacobian_func(v)            # shape (m, n)

        # Compute Huber weights (elementwise, differentiable)
        abs_r = torch.abs(r)
        huber_w = torch.where(abs_r <= huber_delta, torch.ones_like(r), huber_delta / (abs_r + 1e-8))
        # huber_w = huber_delta / (torch.sqrt(r**2 + huber_delta**2) + 1e-8)
        huber_w.detach()

        # Multiply the diagonal elements of W by huber weights
        W_huber = W.clone()
        W_huber_diag = torch.diagonal(W_huber) * huber_w
        W_huber = torch.diag(W_huber_diag)  # rebuild diagonal matrix

        # Apply weighted residuals and Jacobian
        r_w = W_huber @ r
        H_w = W_huber @ H

        # Solve normal equations: (H^T H) delta = -H^T r
        sol = torch.linalg.lstsq(H_w, -r_w)
        delta = sol.solution           # (n,)

        v_new = v + delta
        
        if delta.norm() < tol:
            v = v_new
            break

        v = v_new
        
    return v

def calculate_dop_torch(position: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> dict:
    """
    Calculate Dilution of Precision (DOP) from geometry matrix and weights.

    Parameters
    ----------
    position : torch.Tensor, shape (3,)
        Receiver ECEF position (x, y, z).
    H : torch.Tensor, shape (n_sats, 4)
        Geometry (Jacobian) matrix.
    W : torch.Tensor, shape (n_sats, n_sats)
        Weight matrix (diagonal).

    Returns
    -------
    dict
        A dictionary containing PDOP, TDOP, GDOP, HDOP, and VDOP values.
    """
    n_sats = H.shape[0]
    if n_sats < 4:
        print("Warning: Not enough satellites for DOP calculation")
        return {'pdop': 999.9, 'tdop': 999.9, 'gdop': 999.9, 'hdop': 999.9, 'vdop': 999.9}

    try:
        cov_matrix = torch.linalg.inv(H.T @ W @ H)
    except torch.linalg.LinAlgError:
        print("Warning: Singular matrix in DOP calculation")
        return {'pdop': 999.9, 'tdop': 999.9, 'gdop': 999.9, 'hdop': 999.9, 'vdop': 999.9}

    sigma_x2, sigma_y2, sigma_z2, sigma_t2 = (
        cov_matrix[0, 0], cov_matrix[1, 1], 
        cov_matrix[2, 2], cov_matrix[3, 3]
    )

    pdop = torch.sqrt(sigma_x2 + sigma_y2 + sigma_z2)
    tdop = torch.sqrt(sigma_t2)
    gdop = torch.sqrt(sigma_x2 + sigma_y2 + sigma_z2 + sigma_t2)

    # Warning, breaks autograd - but so far isn't used in a way that requires gradients
    ecef_to_enu = torch.tensor(coords.ecef_to_enu_rot(position.detach().numpy()), dtype=torch.float64)
    pos_cov = cov_matrix[0:3, 0:3]
    enu_cov = ecef_to_enu @ pos_cov @ ecef_to_enu.T

    hdop = torch.sqrt(enu_cov[0, 0] + enu_cov[1, 1])
    vdop = torch.sqrt(enu_cov[2, 2])

    return {
        'pdop': pdop.item(),
        'tdop': tdop.item(),
        'gdop': gdop.item(),
        'hdop': hdop.item(),
        'vdop': vdop.item()
    }

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
        Pseudoranges.
    prr : torch.Tensor, shape (n_sats,)
        Pseudorange rates.
    sat_pos : torch.Tensor, shape (n_sats, 3)
        Satellite positions.
    sat_vel : torch.Tensor, shape (n_sats, 3)
        Satellite velocities.
    prev_estimate : dict | None
        Previous estimate containing 'position', 'velocity', 'clock_bias', 'clock_drift'.
    Wx : torch.Tensor | None
        Weight matrix for position, shape (n_sats, n_sats).
    Wv : torch.Tensor | None
        Weight matrix for velocity, shape (n_sats, n_sats).

    Returns
    -------
    dict
        A dictionary containing the new position, velocity, clock bias, clock drift, and DOP values.
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
        huber_delta=50.0
    )

    if torch.isnan(pos_output).any():
        print("Warning: Least squares returned NaN for position. Returning previous estimate")
        return prev_estimate
    
    
    # Compute DOP
    H_pos = jacobian_residuals_torch(pos_output, sat_pos)
    dop = calculate_dop_torch(pos_output[:3], H_pos, Wx)
    if dop["pdop"] > 10.0:
        print(f"Warning: poor quality solution: PDOP {dop['pdop']:.1f}")
        # return prev_estimate

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
        huber_delta=5.0
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
        "dop": dop,
    }

    return new_pos

def estimate_clock_bias_torch(
    x_true: torch.Tensor,
    xsat: torch.Tensor,
    pr: torch.Tensor
) -> torch.Tensor:
    """
    Estimate clock bias (c * delta_t) in metres via weighted pseudoinverse projection using truth geometry (torch version).

    Parameters
    ----------
    x_true: torch.Tensor, shape (4,)
        True state vector [x, y, z, clock_bias].
    xsat: torch.Tensor, shape (n_sat, 3)
        Satellite positions.
    pr: torch.Tensor, shape (n_sat,)
        Measured pseudoranges.

    Returns
    -------
    torch.Tensor
        Estimated clock bias in metres.
    """
    W = torch.eye(pr.shape[0], dtype=pr.dtype, device=pr.device)
    # Ensure we only use true position
    x_pos = x_true[:3]

    # Line-of-sight and geometric range
    u, rng = los_vector_torch(x_pos, xsat)

    # Apply Earth rotation (Sagnac)
    rng = rng + EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
        xsat[:, 0] * x_pos[1] - xsat[:, 1] * x_pos[0]
    )

    # Pseudorange error (measured - true geometry)
    delta_rho = pr - rng

    H = jacobian_residuals_torch(x_true, xsat)

    H_w = W @ H
    HTH = H_w.T @ H_w

    # Weighted pseudoinverse
    H_pinv = torch.linalg.solve(HTH, H_w.T @ W) # (4, m)

    # Final row corresponds to clock
    p_delta_t = H_pinv[3]  # (m,)

    delta_clock_m = p_delta_t @ delta_rho

    return delta_clock_m

def estimate_clock_drift_torch(
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    xsat: torch.Tensor,
    vsat: torch.Tensor,
    prr: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate clock drift (c * delta_t_dot) in metres per second
    via weighted pseudoinverse projection using truth geometry.
    
    Parameters
    ----------
    x_true: torch.Tensor, shape (4,)
        True state vector [x, y, z, clock_bias].
    v_true: torch.Tensor, shape (4,)
        True velocity vector [vx, vy, vz, clock_drift].
    xsat: torch.Tensor, shape (n_sat, 3)
        Satellite positions.
    vsat: torch.Tensor, shape (n_sat, 3)
        Satellite velocities.
    prr: torch.Tensor, shape (n_sat,)
        Measured pseudorange rates.

    Returns
    -------
    torch.Tensor
        Estimated clock drift in metres per second.
    """

    m = prr.shape[0]
    W = torch.eye(m, dtype=prr.dtype, device=prr.device)

    # True LOS vectors
    u, _ = los_vector_torch(x_true[:3], xsat)

    # True geometric range rate
    rate = torch.zeros(m, dtype=prr.dtype, device=prr.device)

    for i in range(m):
        rate[i] = torch.dot(vsat[i, :3] - v_true[:3], u[i])
        rate[i] += EARTH_ROTATION_SPEED / SPEED_OF_LIGHT * (
            vsat[i, 1] * x_true[0]
            + xsat[i, 1] * v_true[0]
            - vsat[i, 0] * x_true[1]
            - xsat[i, 0] * v_true[1]
        )

    # Pseudorange rate error (measured - true geometry)
    delta_rho_dot = prr - rate

    # PRR Jacobian wrt velocity state
    H = torch.zeros((m, 4), dtype=prr.dtype, device=prr.device)
    H[:, :3] = -u
    H[:, 3] = 1.0

    # Weighted pseudoinverse
    H_w = W @ H
    HTH = H_w.T @ H_w
    H_pinv = torch.linalg.solve(HTH, H_w.T @ W)

    # Clock drift row
    p_delta_t_dot = H_pinv[3]

    delta_clock_drift_mps = p_delta_t_dot @ delta_rho_dot

    return delta_clock_drift_mps


class Kalman_filter:
    """
    GNSS loosely coupled Kalman filter for position, velocity, clock bias, and clock drift estimation.
    """

    def __init__(
        self,
        # Process noise
        sigma_acc: float = 2.0,
        sigma_b: float = 30.0,
        sigma_d: float = 0.5,
        # Measurement noise
        sigma_p_h: float = 10.0,
        sigma_p_v: float = 25.0, # larger than horizontal, GNSS geometry is worse vertically
        sigma_v: float = 0.5,
        # Initial uncertainty
        sigma_p0: float = 50.0,
        sigma_v0: float = 10.0,
        sigma_b0: float = 200.0,
        sigma_d0: float = 2.0,
        # Adaptive parameters
        min_speed_threshold: float = 0.5,
        max_speed: float = 50.0, # (~180 km/h) - upper limit for cars
    ):
        """
        Initialise the Kalman filter with given noise parameters and initial uncertainties.

        Parameters
        ----------
        sigma_acc : float
            Acceleration process noise standard deviation (m/s^2).
        sigma_b : float
            Clock bias process noise standard deviation (m).
        sigma_d : float
            Clock drift process noise standard deviation (m/s).
        sigma_p_h : float
            Horizontal position measurement noise standard deviation (m).
        sigma_p_v : float
            Vertical position measurement noise standard deviation (m).
        sigma_v : float
            Velocity measurement noise standard deviation (m/s).
        sigma_p0 : float
            Initial position uncertainty standard deviation (m).
        sigma_v0 : float
            Initial velocity uncertainty standard deviation (m/s).
        sigma_b0 : float
            Initial clock bias uncertainty standard deviation (m).
        sigma_d0 : float
            Initial clock drift uncertainty standard deviation (m/s).
        min_speed_threshold : float
            Minimum speed threshold to detect stationary state (m/s).
        max_speed : float
            Maximum expected speed for adaptive process noise scaling (m/s).
        """
        self.initialised_ = False
        self.min_speed_threshold_ = min_speed_threshold
        self.max_speed_ = max_speed
        self.stationary_count_ = 0
        self.sigma_acc_ = sigma_acc
        self.sigma_b_ = sigma_b
        self.sigma_d_ = sigma_d
        self.sigma_p_h_ = sigma_p_h
        self.sigma_p_v_ = sigma_p_v
 
        # State transition F (8x8)
        self.F_ = np.eye(8)
 
        # Observation H (8x8) - measuring state directly
        self.H_ = np.eye(8)
 
        # Process noise Q (8x8)
        self.Q_ = np.zeros((8, 8))
 
        # Measurement noise R (8x8)
        # Position block (0:3, 0:3) is built fresh each update() call via
        # _enu_noise_to_ecef, so only velocity and clock terms are set here
        self.R_ = np.zeros((8, 8))
        for i in range(3):
            self.R_[i + 3, i + 3] = sigma_v * sigma_v
        self.R_[6, 6] = sigma_b * sigma_b
        self.R_[7, 7] = sigma_d * sigma_d
 
        # Initial state
        self.x_ = np.zeros((8, 1))
 
        # Initial covariance P (8x8)
        self.P_ = np.zeros((8, 8))
        for i in range(3):
            self.P_[i, i] = sigma_p0 * sigma_p0
            self.P_[i + 3, i + 3] = sigma_v0 * sigma_v0
        self.P_[6, 6] = sigma_b0 * sigma_b0
        self.P_[7, 7] = sigma_d0 * sigma_d0
 
    def _enu_noise_to_ecef(self, sigma_h: float, sigma_v: float, ecef_pos: np.ndarray) -> np.ndarray:
        """
        Convert noise from ENU to ECEF coordinates.

        Parameters
        ----------
        sigma_h : float
            Horizontal noise standard deviation (m).
        sigma_v : float
            Vertical noise standard deviation (m).
        ecef_pos : np.ndarray, shape (3,)
            Receiver position in ECEF coordinates (x, y, z).

        Returns
        -------
        np.ndarray
            3x3 covariance matrix in ECEF coordinates.
        """
        # Rotation matrix from ENU to ECEF at this position
        R_enu_to_ecef = coords.enu_to_ecef_rot(ecef_pos)
 
        # Diagonal covariance in ENU: East, North, Up
        cov_enu = np.diag([sigma_h**2, sigma_h**2, sigma_v**2])
 
        # Rotate to ECEF: cov_ecef = R * cov_enu * R^T
        return R_enu_to_ecef @ cov_enu @ R_enu_to_ecef.T
 
    def initialise(self, pos: np.ndarray, vel: np.ndarray, clock_bias: float, clock_drift: float) -> None:
        """
        Initialise internal state matricies with given position, velocity, clock bias, and clock drift.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Initial position in ECEF coordinates (x, y, z).
        vel : np.ndarray, shape (3,)
            Initial velocity in ECEF coordinates (vx, vy, vz).
        clock_bias : float
            Initial clock bias.
        clock_drift : float
            Initial clock drift.

        Returns
        -------
        None
        """
        self.x_[0:3, 0] = pos
        self.x_[3:6, 0] = vel
        self.x_[6, 0] = clock_bias
        self.x_[7, 0] = clock_drift
        self.stationary_count_ = 0
        self.initialised_ = True
 
    def predict(self, dt: float, current_speed: float | None = None) -> None:
        """
        Predict the next state of the system.

        Parameters
        ----------
        dt : float
            Time step (s).
        current_speed : float, optional
            Current speed (m/s).

        Returns
        -------
        None
        """
        if not self.initialised_:
            return
 
        I3 = np.eye(3)
 
        self.F_.fill(0)
        np.fill_diagonal(self.F_, 1.0)
        self.F_[0:3, 3:6] = dt * I3  # position depends on velocity
        self.F_[6, 7] = dt            # clock bias depends on drift
 
        q11 = (dt**4) / 4.0 * I3
        q12 = (dt**3) / 2.0 * I3
        q22 = (dt**2) * I3
 
        self.Q_.fill(0)
        self.Q_[0:3, 0:3] = q11
        self.Q_[0:3, 3:6] = q12
        self.Q_[3:6, 0:3] = q12
        self.Q_[3:6, 3:6] = q22
        self.Q_[0:6, 0:6] *= (self.sigma_acc_ * self.sigma_acc_)
 
        self.Q_[6, 6] = self.sigma_b_ * self.sigma_b_ * dt
        self.Q_[7, 7] = self.sigma_d_ * self.sigma_d_ * dt
 
        Q = self.Q_.copy()
 
        if current_speed is not None:
            if current_speed < self.min_speed_threshold_:
                self.stationary_count_ += 1
                if self.stationary_count_ > 3:  # ~3 seconds stopped
                    Q[0:6, 0:6] *= 0.1
            else:
                self.stationary_count_ = 0
                speed_factor = min(current_speed / 20.0, 2.0)
                Q[0:6, 0:6] *= speed_factor
 
        self.x_ = self.F_.dot(self.x_)
        self.P_ = self.F_.dot(self.P_).dot(self.F_.T) + Q
 
    def update(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        clock_bias: float,
        clock_drift: float,
        hdop: float | None = None,
        vdop: float | None = None,
        satellite_count: int | None = None,
        speed: float | None = None,
    ) -> None:
        """
        Update the Kalman filter with a new position, velocity, clock bias, and clock drift measurement.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Measured position in ECEF coordinates (x, y, z).
        vel : np.ndarray, shape (3,)
            Measured velocity in ECEF coordinates (vx, vy, vz).
        clock_bias : float
            Measured clock bias.
        clock_drift : float
            Measured clock drift.
        hdop : float, optional
            Horizontal Dilution of Precision.
        vdop : float, optional
            Vertical Dilution of Precision.
        satellite_count : int, optional
            Number of satellites used in the measurement.
        speed : float, optional
            Current speed (m/s).

        Returns
        -------
        None
        """
        if not self.initialised_:
            self.initialise(pos, vel, clock_bias, clock_drift)
            return
 
        # Velocity constraint for cars
        if speed is not None and speed > self.max_speed_:
            return
 
        R = self.R_.copy()
 
        # Compute sigma values before rotation, applying DOP and satellite count scaling
        sigma_h = self.sigma_p_h_
        sigma_v = self.sigma_p_v_
 
        if hdop is not None:
            sigma_h *= max(hdop / 2.0, 1.0)  # baseline HDOP=2
 
        if vdop is not None:
            sigma_v *= max(vdop / 2.0, 1.0)  # baseline VDOP=2
 
        if satellite_count is not None and satellite_count < 6:
            # Inflate variance by 2x, so sigma by sqrt(2)
            sigma_h *= np.sqrt(2.0)
            sigma_v *= np.sqrt(2.0)
 
        # Build position covariance in ENU and rotate to ECEF so that
        # horizontal/vertical noise is correctly oriented at this location
        R[0:3, 0:3] = self._enu_noise_to_ecef(sigma_h, sigma_v, pos)
 
        z = np.zeros((8, 1))
        z[0:3, 0] = pos
        z[3:6, 0] = vel
        z[6, 0] = clock_bias
        z[7, 0] = clock_drift
 
        y = z - self.H_.dot(self.x_)
 
        S = self.H_.dot(self.P_).dot(self.H_.T) + R
 
        K = np.linalg.solve(S.T, self.H_.dot(self.P_).T).T
 
        self.x_ = self.x_ + K.dot(y)
 
        I = np.eye(8)
        I_KH = I - K.dot(self.H_)
        self.P_ = I_KH.dot(self.P_).dot(I_KH.T) + K.dot(R).dot(K.T)
 
    def get_position(self):
        """
        Get the current estimated position from the Kalman filter state.

        Returns
        -------
        np.ndarray
            Estimated position in ECEF coordinates (x, y, z).
        """
        return self.x_[0:3, 0].copy()
 
    def get_velocity(self):
        """
        Get the current estimated velocity from the Kalman filter state.

        Returns
        -------
        np.ndarray
            Estimated velocity in ECEF coordinates (vx, vy, vz).
        """
        return self.x_[3:6, 0].copy()
 
    def get_clock_bias(self):
        """
        Get the current estimated clock bias from the Kalman filter state.

        Returns
        -------
        float
            Estimated clock bias.
        """
        return float(self.x_[6, 0])
 
    def get_clock_drift(self):
        """
        Get the current estimated clock drift from the Kalman filter state.

        Returns
        -------
        float
            Estimated clock drift.
        """
        return float(self.x_[7, 0])
    