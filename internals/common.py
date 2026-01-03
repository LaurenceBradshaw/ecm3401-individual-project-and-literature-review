import pandas as pd
import torch
from internals.coord_systems import lla_to_ecef, heading_speed_to_ecef
import internals.gnss_positioning as gp

PR_COL = 'CorrectedPseudorange'
PRR_COL = 'CorrectedPseudorangeRateMetersPerSecond'
SAT_POS_COLS = ['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']
SAT_VEL_COLS = ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']

device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_device() -> torch.device:
    return device_

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
    pos_truth = torch.tensor(pos_truth, dtype=torch.float32, device=get_device())
    vel_truth = torch.tensor(vel_truth, dtype=torch.float32, device=get_device())
    return pos_truth, vel_truth

def compute_pos(epoch_df: pd.DataFrame, pr_weights: torch.Tensor, pr_correction: torch.Tensor, prr_weights: torch.Tensor, curr_pos: dict) -> dict:
    # Grab the required columns and convert to tensors
    pr = torch.tensor(epoch_df[PR_COL].to_numpy(), dtype=torch.float32, device=get_device())
    if pr_correction is not None:
        pr = pr + pr_correction
        
    prr = torch.tensor(epoch_df[PRR_COL].to_numpy(), dtype=torch.float32, device=get_device())
    sat_pos = torch.tensor(epoch_df[SAT_POS_COLS].to_numpy(), dtype=torch.float32, device=get_device())
    sat_vel = torch.tensor(epoch_df[SAT_VEL_COLS].to_numpy(), dtype=torch.float32, device=get_device())
    # Compute updated position estimate
    curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights, Wv=prr_weights, prev_estimate=curr_pos)
    return curr_pos