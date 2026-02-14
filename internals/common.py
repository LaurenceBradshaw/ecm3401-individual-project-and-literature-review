import pandas as pd
import numpy as np
import torch
from internals.coord_systems import lla_to_ecef, heading_speed_to_ecef
from internals.constants import PR_COL, PRR_COL, SAT_POS_COLS, SAT_VEL_COLS
import internals.gnss_positioning as gp

device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_device() -> torch.device:
    # return device_
    return "cpu" # Tried to use GPU, but only my laptop has an nvidia gpu and its slower than my main desktop cpu.

def get_ground_truth(truth_df: pd.DataFrame, epoch_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    # Find closest ground truth row by time
    gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
    # Convert to ECEF
    pos_truth = lla_to_ecef(
        gt_row[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy().flatten()
    )
    # Convert velocity to ECEF
    vel_truth = heading_speed_to_ecef(
        gt_row['BearingDegrees'], gt_row['SpeedMps'],
        gt_row['LatitudeDegrees'], gt_row['LongitudeDegrees']
    )
    # Convert to tensors
    pos_truth = torch.tensor(pos_truth, dtype=torch.float64, device=get_device())
    vel_truth = torch.tensor(vel_truth, dtype=torch.float64, device=get_device())
    return pos_truth, vel_truth

def compute_pos_torch(epoch_df: pd.DataFrame, pr_weights: torch.Tensor | None, pr_correction: torch.Tensor | None, prr_weights: torch.Tensor | None, curr_pos: dict | None) -> dict:
    # Grab the required columns and convert to tensors
    pr = torch.tensor(epoch_df[PR_COL].to_numpy(), dtype=torch.float64, device=get_device())
    if pr_correction is not None:
        pr = pr + pr_correction
        
    prr = torch.tensor(epoch_df[PRR_COL].to_numpy(), dtype=torch.float64, device=get_device())
    sat_pos = torch.tensor(epoch_df[SAT_POS_COLS].to_numpy(), dtype=torch.float64, device=get_device())
    sat_vel = torch.tensor(epoch_df[SAT_VEL_COLS].to_numpy(), dtype=torch.float64, device=get_device())
    # Compute updated position estimate
    curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights, Wv=prr_weights, prev_estimate=curr_pos)
    return curr_pos

def compute_pos(epoch_df: pd.DataFrame, pr_weights: np.ndarray | None, pr_correction: np.ndarray | None, prr_weights: np.ndarray | None, curr_pos: dict | None) -> dict:
    # Extract columns once, no torch, no devices
    pr = epoch_df[PR_COL].to_numpy(dtype=np.float64)
    if pr_correction is not None:
        pr = pr + pr_correction

    prr = epoch_df[PRR_COL].to_numpy(dtype=np.float64)
    sat_pos = epoch_df[SAT_POS_COLS].to_numpy(dtype=np.float64)
    sat_vel = epoch_df[SAT_VEL_COLS].to_numpy(dtype=np.float64)

    curr_pos = gp.position(pr,prr,sat_pos,sat_vel,Wx=pr_weights,Wv=prr_weights,prev_estimate=curr_pos)

    return curr_pos

def compute_residual_matrix(epoch_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_sats = len(epoch_df)

    pr_residual_matrix = np.zeros(
        (n_sats, n_sats),
        dtype=np.float64,
    )

    prr_residual_matrix = np.zeros(
        (n_sats, n_sats),
        dtype=np.float64,
    )

    # Ensure unique satellite + signal identity
    if epoch_df.duplicated(subset=["ConstellationType", "Svid", "SignalType"]).any():
        raise ValueError("Duplicate satellite entries with the same signal type found in epoch_df.")

    # Pre-extract identity for stable indexing
    sat_keys = list(
        zip(epoch_df["ConstellationType"], epoch_df["Svid"], epoch_df["SignalType"])
    )

    sat_num = 0
    for row_idx, sat_row in epoch_df.iterrows():
        excluded_key = (sat_row["ConstellationType"], sat_row["Svid"], sat_row["SignalType"])
        included_mask = [key != excluded_key for key in sat_keys]
        included_sats = epoch_df[included_mask]

        curr_pos = compute_pos(included_sats, None, None, None, curr_pos=None)

        pr = included_sats[PR_COL].to_numpy()
        prr = included_sats[PRR_COL].to_numpy()
        sat_pos = included_sats[SAT_POS_COLS].to_numpy()
        sat_vel = included_sats[SAT_VEL_COLS].to_numpy()

        x = np.zeros(4)
        x[:3] = curr_pos["position"]
        x[3] = curr_pos["clock_bias"]

        v = np.zeros(4)
        v[:3] = curr_pos["velocity"]
        v[3] = curr_pos["clock_drift"]

        # Residuals for included satellites only (length N-1)
        pr_res = gp.pr_residuals(x, sat_pos, pr)
        prr_res = gp.prr_residuals(v, sat_vel, prr, x, sat_pos)

        # Insert into full row
        col_indices = [
            i for i, key in enumerate(sat_keys) if key != excluded_key
        ]
        pr_residual_matrix[sat_num, col_indices] = pr_res.astype(np.float64)
        prr_residual_matrix[sat_num, col_indices] = prr_res.astype(np.float64)

        # Diagonal explicitly set
        pr_residual_matrix[sat_num, sat_num] = 0.0
        prr_residual_matrix[sat_num, sat_num] = 0.0
        sat_num += 1

    return pr_residual_matrix, prr_residual_matrix
