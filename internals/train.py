import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from internals.coord_systems import lla_to_ecef, ecef_to_lla, heading_speed_to_ecef
from internals import gnss_positioning as gp
import internals.common as common

PR_COL = 'CorrectedPseudorange'
PRR_COL = 'CorrectedPseudorangeRateMetersPerSecond'
SAT_POS_COLS = ['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']
SAT_VEL_COLS = ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']

def setup(base_path: str, network_cls: torch.nn.Module) -> None:
    global net, optimizer, features
    ###############
    features = network_cls.features

    dataset_stats_file = f"{os.path.dirname(base_path)}/dataset_stats.csv"
    stats_df = pd.read_csv(dataset_stats_file)

    mean = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'mean'].values[0] for col in features
    ], dtype=torch.float32).to(common.get_device())
    std = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'std_dev'].values[0] for col in features
    ], dtype=torch.float32).to(common.get_device())

    net = network_cls(mean=mean, std=std, feat_dim=len(features)).to(common.get_device())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def run(df: pd.DataFrame, truth_df: pd.DataFrame, get_feats: callable, save_path: str) -> None:
    N = df['epoch_id'].nunique()

    # Initial position estimate using first epoch otherwise gradients are terrible initially
    # This doesn't need any weights, the initial position is just to get a reasonable starting point
    curr_pos = common.compute_pos(df[df['epoch_id'] == 0], pr_weights=None, pr_correction=None, prr_weights=None, curr_pos=None)

    optimizer.zero_grad()
    for epoch_id, epoch_df in df.groupby("epoch_id"):
        pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)
        feats = get_feats(epoch_df, features, common.get_device())

        pr_weights, pr_error, prr_weights = net(*feats)

        # assert all weights are not 1
        assert not torch.all(pr_weights == 1.0), "All predicted pseudorange weights are 1.0"

        # TODO: do this before outputting from the network.
        Wx = torch.diag_embed(pr_weights.squeeze(1))
        pr_error = pr_error.T.squeeze(0)
        Wv = torch.diag_embed(prr_weights.squeeze(1))

        # Get the baseline weights to compare model against
        Wx_baseline = torch.diag(torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float32, device='cpu'))
        Wv_baseline = torch.diag(torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float32, device='cpu'))

        # Update position estimate with weighted least squares - both model and baseline
        baseline_pos = common.compute_pos(epoch_df, pr_weights=Wx_baseline, pr_correction=None, prr_weights=Wv_baseline, curr_pos=curr_pos)
        curr_pos = common.compute_pos(epoch_df, pr_weights=Wx, pr_correction=pr_error, prr_weights=Wv, curr_pos=curr_pos)

        # Compute losses
        pr_baseline_loss = torch.linalg.norm(baseline_pos['position'] - pos_truth)
        prr_baseline_loss = torch.linalg.norm(baseline_pos['velocity'] - vel_truth)
        pr_loss = torch.linalg.norm(curr_pos['position'] - pos_truth)
        prr_loss = torch.linalg.norm(curr_pos['velocity'] - vel_truth)
        # Scale losses by baseline to get relative improvement
        pos_loss = pr_loss / (pr_baseline_loss.detach().item() + 1e-6)
        vel_loss = prr_loss / (prr_baseline_loss.detach().item() + 1e-6)
        # Penalise predicted weights that deviate too far from 0.5 mean and have low variance
        # Otherwise the network predicted weights all collapse to 1, and through sigmoid backwards, this results in no gradients
        pr_weight_reg = ((pr_weights.mean() - 0.5)**2) - 0.1*pr_weights.var()
        prr_weight_reg = ((prr_weights.mean() - 0.5)**2) - 0.5*prr_weights.var()

        combined_loss = (
            pr_loss
            + 2.5 * prr_loss
            # + 1 * torch.relu(pr_loss - pr_baseline_loss)
            # + 2.5 * torch.relu(prr_loss - prr_baseline_loss)
            # + 50 * pr_weight_reg**2 + 100 * prr_weight_reg**2
        )

        pos_gain = pr_baseline_loss.detach().item() - pr_loss.detach().item()
        vel_gain = prr_baseline_loss.detach().item() - prr_loss.detach().item()
        print(
            f"Epoch {epoch_id + 1} / {N} | "
            f"Pos err: {pr_loss.item():.3f} m (baseline {pr_baseline_loss.item():.3f}, Δ{pos_gain:.3f}) | "
            f"Vel err: {prr_loss.item():.3f} m/s (baseline {prr_baseline_loss.item():.3f}, Δ{vel_gain:.3f}) | "
            f"Combined Loss: {combined_loss.item():.3e}"
        )

        # Simulate batching by accumulating gradients over a number of epochs
        # It's easier to implement this way rather than actually batching epochs due to variable satellite counts
        # otherwise would need to pad inputs and handle masks
        (combined_loss / 32).backward()
        if epoch_id % 32 == 0 or epoch_id == df['epoch_id'].unique().max():
            optimizer.step()
            optimizer.zero_grad()

        # Detach the current position state for the next epoch
        curr_pos = {
            'position': curr_pos['position'].detach(),
            'velocity': curr_pos['velocity'].detach(),
            'clock_bias': curr_pos['clock_bias'].detach(),
            'clock_drift': curr_pos['clock_drift'].detach()
        }

    # Save the model after each file
    torch.save(net.state_dict(), save_path)
