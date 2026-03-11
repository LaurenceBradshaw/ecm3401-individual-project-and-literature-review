import pandas as pd
import os
import torch
from internals import gnss_positioning as gp
from internals.constants import PR_COL, PRR_COL, SAT_POS_COLS, SAT_VEL_COLS
import internals.common as common

def colour_delta(val, max_val=1e1):
    # Clamp magnitude to max_val
    magnitude = min(abs(val), max_val)
    # Scale intensity 0-255
    intensity = int(30 + 225 * (magnitude / max_val))  # 30->255
    if val > 0:
        # Green: ANSI 38;2;r;g;b, using r=0, b=0, g=intensity
        return f"\033[38;2;0;{intensity};0m{val:+.3e}\033[0m"
    elif val < 0:
        # Red: r=intensity, g=0, b=0
        return f"\033[38;2;{intensity};0;0m{val:+.3e}\033[0m"
    else:
        return f"{val:+.3e}"  # no colour for zero

def setup(base_path: str, network_cls: torch.nn.Module, compute_baseline: bool) -> None:
    global net, optimizer, features, scheduler, _compute_baseline
    ###############
    features = network_cls.features

    _compute_baseline = compute_baseline

    dataset_stats_file = f"{base_path}/dataset_stats.csv"
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
        gamma=0.05
    )

def step_scheduler():
    scheduler.step()

def run(df: pd.DataFrame, truth_df: pd.DataFrame, get_feats: callable, save_path: str) -> None:
    N = df['epoch_id'].nunique()

    # Initial position estimate using first epoch otherwise gradients are terrible initially
    # This doesn't need any weights, the initial position is just to get a reasonable starting point
    epoch_start = df['epoch_id'].min()
    curr_pos = common.compute_pos_torch(df[df['epoch_id'] == epoch_start], pr_weights=None, pr_correction=None, prr_weights=None, prr_correction=None, curr_pos=None)
    baseline_pos = curr_pos.copy()

    optimizer.zero_grad()
    epoch_num = 1
    for epoch_id, epoch_df in df.groupby("epoch_id"):
        pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)
        feats = get_feats(epoch_df, features, common.get_device())

        Wx, pr_errors, Wv, prr_errors = net(*feats)

        if _compute_baseline:
            # Get the baseline weights to compare model against
            Wx_baseline = torch.diag(torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float64, device='cpu')) # TODO: Test against inverse measurement uncertainty.
            Wv_baseline = torch.diag(torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float64, device='cpu'))

            # Update position estimate with weighted least squares - both model and baseline
            baseline_pos = common.compute_pos_torch(epoch_df, pr_weights=Wx_baseline, pr_correction=None, prr_weights=Wv_baseline, prr_correction=None, curr_pos=baseline_pos)

            pr_baseline = torch.concat([baseline_pos['position'], baseline_pos['clock_bias'].unsqueeze(0)])
            prr_baseline = torch.concat([baseline_pos['velocity'], baseline_pos['clock_drift'].unsqueeze(0)])

            pr_baseline_loss = torch.linalg.norm(pr_baseline - pos_truth)
            prr_baseline_loss = torch.linalg.norm(prr_baseline - vel_truth)

        curr_pos = common.compute_pos_torch(epoch_df, pr_weights=Wx, pr_correction=pr_errors, prr_weights=Wv, prr_correction=prr_errors, curr_pos=curr_pos)

        pr_model = torch.concat([curr_pos['position'], curr_pos['clock_bias'].unsqueeze(0)])
        prr_model = torch.concat([curr_pos['velocity'], curr_pos['clock_drift'].unsqueeze(0)])

        pr_loss = torch.linalg.norm(pr_model - pos_truth)
        prr_loss = torch.linalg.norm(prr_model - vel_truth)

        combined_loss = 10 * pr_loss + 30 * prr_loss

        epoch_pad = ' '*(len(str(N)) - len(str(epoch_num)))
        if _compute_baseline:
            pos_gain = pr_baseline_loss.detach().item() - pr_loss.detach().item()
            vel_gain = prr_baseline_loss.detach().item() - prr_loss.detach().item()
            print(
                f"GNSS Epoch {epoch_pad}{epoch_num} / {N} | "
                f"Pos err: {pr_loss.item():.3e} m (baseline {pr_baseline_loss.item():.3e}, Δ{colour_delta(pos_gain, 1e1)}) | "
                f"Vel err: {prr_loss.item():.3e} m/s (baseline {prr_baseline_loss.item():.3e}, Δ{colour_delta(vel_gain, 0.5)}) | "
                f"Combined Loss: {combined_loss.item():.3e}"
            )
        else:
            print(
                f"GNSS Epoch {epoch_pad}{epoch_num} / {N} | "
                f"Pos err: {pr_loss.item():.3e} m | "
                f"Vel err: {prr_loss.item():.3e} m/s | "
                f"Combined Loss: {combined_loss.item():.3e}"
            )
        epoch_num += 1

        # Simulate batching by accumulating gradients over a number of epochs
        # It's easier to implement this way rather than actually batching epochs due to variable satellite counts
        # otherwise would need to pad inputs and handle masks
        (combined_loss / 20).backward()
        if epoch_id % 20 == 0 or epoch_id == df['epoch_id'].unique().max():
            optimizer.step()
            optimizer.zero_grad()

        # Detach the current position state for the next epoch
        curr_pos = {
            'position': curr_pos['position'].detach(),
            'velocity': curr_pos['velocity'].detach(),
            'clock_bias': curr_pos['clock_bias'].detach(),
            'clock_drift': curr_pos['clock_drift'].detach()
        }
        if _compute_baseline:
            baseline_pos = {
                'position': baseline_pos['position'].detach(),
                'velocity': baseline_pos['velocity'].detach(),
                'clock_bias': baseline_pos['clock_bias'].detach(),
                'clock_drift': baseline_pos['clock_drift'].detach()
            }

    # Save the model after each file
    torch.save(net.state_dict(), save_path)
