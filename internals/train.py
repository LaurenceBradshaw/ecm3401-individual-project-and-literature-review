import pandas as pd
import os
import torch
from internals import gnss_positioning as gp
from internals.constants import PR_COL, PRR_COL, SAT_POS_COLS, SAT_VEL_COLS
import internals.common as common

TRAINING_DRIVES = [
    os.path.join("2020-06-25-00-34-us-ca-mtv-sb-101", "pixel4xl"), 
    os.path.join("2022-02-24-18-29-us-ca-lax-o", "mi8"),
    os.path.join("2023-09-06-00-01-us-ca-routen", "pixel4xl"),
    os.path.join("2023-03-08-21-34-us-ca-mtv-u", "pixel7pro"),
    os.path.join("2023-05-24-20-26-us-ca-sjc-ge2", "pixel7pro"),
    os.path.join("2021-03-10-23-13-us-ca-mtv-h", "mi8"),
    os.path.join("2021-07-27-19-49-us-ca-mtv-b", "mi8"),
    os.path.join("2021-08-04-20-40-us-ca-sjc-c", "sm-g988b"),
]

def setup(base_path: str, network_cls: torch.nn.Module) -> None:
    global net, optimizer, features, scheduler
    ###############
    features = network_cls.features

    dataset_stats_file = f"{os.path.dirname(base_path)}/dataset_stats.csv"
    stats_df = pd.read_csv(dataset_stats_file)

    mean = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'mean'].values[0] for col in features
    ], dtype=torch.float64).to(common.get_device())
    std = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'std_dev'].values[0] for col in features
    ], dtype=torch.float64).to(common.get_device())

    net = network_cls(mean=mean, std=std, feat_dim=len(features)).to(common.get_device())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

def run(df: pd.DataFrame, truth_df: pd.DataFrame, get_feats: callable, save_path: str) -> None:
    N = df['epoch_id'].nunique()

    # Initial position estimate using first epoch otherwise gradients are terrible initially
    # This doesn't need any weights, the initial position is just to get a reasonable starting point
    epoch_start = df['epoch_id'].min()
    curr_pos = common.compute_pos_torch(df[df['epoch_id'] == epoch_start], pr_weights=None, pr_correction=None, prr_weights=None, curr_pos=None)

    optimizer.zero_grad()
    epoch_num = 1
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
        Wx_baseline = torch.diag(torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float64, device='cpu'))
        Wv_baseline = torch.diag(torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float64, device='cpu'))

        # Update position estimate with weighted least squares - both model and baseline
        baseline_pos = common.compute_pos_torch(epoch_df, pr_weights=Wx_baseline, pr_correction=None, prr_weights=Wv_baseline, curr_pos=curr_pos)
        curr_pos = common.compute_pos_torch(epoch_df, pr_weights=Wx, pr_correction=pr_error, prr_weights=Wv, curr_pos=curr_pos)

        # pr_loss = torch.linalg.norm(curr_pos['position'] - pos_truth)
        # pr_baseline_loss = torch.linalg.norm(baseline_pos['position'] - pos_truth)
        # prr_baseline_loss = torch.linalg.norm(baseline_pos['velocity'] - vel_truth) # TODO: estimate clock drift velocity too from truth
        # prr_loss = torch.linalg.norm(curr_pos['velocity'] - vel_truth)

        pr = torch.tensor(epoch_df[PR_COL].to_numpy(), dtype=torch.float64, device=common.get_device())
        prr = torch.tensor(epoch_df[PRR_COL].to_numpy(), dtype=torch.float64, device=common.get_device())
        sat_pos = torch.tensor(epoch_df[SAT_POS_COLS].to_numpy(), dtype=torch.float64, device=common.get_device())
        sat_vel = torch.tensor(epoch_df[SAT_VEL_COLS].to_numpy(), dtype=torch.float64, device=common.get_device())

        # clock_bias_truth, clock_drift_truth = gp.estimate_rx_clock_bias_and_drift_torch(pr, prr, sat_pos, sat_vel, pos_truth, vel_truth)
        pos_truth_vec = torch.concat([pos_truth, torch.zeros(1, dtype=torch.float64, device=common.get_device())])
        clock_bias_truth = gp.estimate_clock_bias_via_pseudoinverse(pos_truth_vec, sat_pos, pr)
        pos_truth = torch.concat([pos_truth, clock_bias_truth.unsqueeze(0)])

        vel_truth_vec = torch.concat([vel_truth, torch.zeros(1, dtype=torch.float64, device=common.get_device())])
        clock_drift_truth = gp.estimate_clock_drift_via_pseudoinverse(pos_truth, vel_truth_vec, sat_pos, sat_vel, prr)
        vel_truth = torch.concat([vel_truth, clock_drift_truth.unsqueeze(0)])

        pr_baseline = torch.concat([baseline_pos['position'], baseline_pos['clock_bias'].unsqueeze(0)])
        pr_model = torch.concat([curr_pos['position'], curr_pos['clock_bias'].unsqueeze(0)])
        prr_baseline = torch.concat([baseline_pos['velocity'], baseline_pos['clock_drift'].unsqueeze(0)])
        prr_model = torch.concat([curr_pos['velocity'], curr_pos['clock_drift'].unsqueeze(0)])

        pr_baseline_loss = torch.linalg.norm(pr_baseline - pos_truth)
        pr_loss = torch.linalg.norm(pr_model - pos_truth)
        prr_baseline_loss = torch.linalg.norm(prr_baseline - vel_truth)
        prr_loss = torch.linalg.norm(prr_model - vel_truth)

        combined_loss = (
            10 * pr_loss
            + 30 * prr_loss
        )

        pos_gain = pr_baseline_loss.detach().item() - pr_loss.detach().item()
        vel_gain = prr_baseline_loss.detach().item() - prr_loss.detach().item()
        pos_pad = ' ' if pos_gain >= 0 else ''
        vel_pad = ' ' if vel_gain >= 0 else ''
        epoch_pad = ' '*(len(str(N)) - len(str(epoch_id + 1)))
        print(
            f"Epoch {epoch_pad}{epoch_num} / {N} | "
            f"Pos err: {pr_loss.item():.3e} m (baseline {pr_baseline_loss.item():.3e}, Δ{pos_pad}{pos_gain:.3e}) | "
            f"Vel err: {prr_loss.item():.3e} m/s (baseline {prr_baseline_loss.item():.3e}, Δ{vel_pad}{vel_gain:.3e}) | "
            f"Combined Loss: {combined_loss.item():.3e}"
        )
        epoch_num += 1

        # Simulate batching by accumulating gradients over a number of epochs
        # It's easier to implement this way rather than actually batching epochs due to variable satellite counts
        # otherwise would need to pad inputs and handle masks
        (combined_loss / 20).backward()
        if epoch_id % 20 == 0 or epoch_id == df['epoch_id'].unique().max():
            optimizer.step()
            scheduler.step()
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
