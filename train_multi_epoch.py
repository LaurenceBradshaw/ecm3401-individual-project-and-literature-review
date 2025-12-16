import pandas as pd
import os
import numpy as np
from data_file_iter import Data_file_iterator
from model import Gnss_multi_epoch_net
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.coord_systems import lla_to_ecef, ecef_to_lla, heading_speed_to_ecef
import utils.gnss_positioning as gp
from epoch_manager import Epoch_manager
from internals.train import setup, run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


# if __name__ == "__main__":
#     base_path = "./smartphone-decimeter-2023/sdc2023"

#     data_iter = Data_file_iterator(base_path, split='train', limit=10)

#     features = Gnss_multi_epoch_net.features

#     dataset_stats_file = f"{base_path}/dataset_stats.csv"
#     stats_df = pd.read_csv(dataset_stats_file)
#     mean = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'mean'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     std = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'std_dev'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     net = Gnss_multi_epoch_net(
#         mean=mean,
#         std=std,
#         feat_dim=len(features)
#     ).to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#     for df, truth_df in data_iter:
#         print(f"Processing file: {data_iter.get_current_file_path()}")

#         curr_pos = None
#         # Initial position estimate using first epoch otherwise gradients are terrible initially
#         pr = df[df['epoch_id'] == 0]['CorrectedPseudorange'].to_numpy()
#         prr = df[df['epoch_id'] == 0]['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
#         sat_pos = df[df['epoch_id'] == 0][['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
#         sat_vel = df[df['epoch_id'] == 0][['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

#         curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)
#         curr_pos["position"] = torch.tensor(curr_pos["position"], dtype=torch.float32)
#         curr_pos["clock_bias"] = torch.tensor(curr_pos["clock_bias"], dtype=torch.float32)
#         curr_pos["velocity"] = torch.tensor(curr_pos["velocity"], dtype=torch.float32)
#         curr_pos["clock_drift"] = torch.tensor(curr_pos["clock_drift"], dtype=torch.float32)

#         optimizer.zero_grad()
#         epoch_manager = Epoch_manager()
#         for epoch_id, epoch_df in df.groupby("epoch_id"):
#             # The instantaneous per-satellite features (num_sats, feat_dim)
#             feats_np = epoch_df[features].to_numpy(dtype=np.float32)
#             feats = torch.tensor(feats_np, dtype=torch.float32, device=device)

#             # ----- 1) update temporal histories -----
#             sat_ids = epoch_df["sat_identifier"].tolist()

#             for sat_id, feat_vec in zip(sat_ids, feats):
#                 epoch_manager.add_entry(sat_id, feat_vec.detach().cpu())

#             # ----- 2) retrieve per-satellite sequences -----
#             history_dict = epoch_manager.batch_history(sat_ids)

#             # Build padded tensor: (num_sats, max_len, feat_dim)
#             sequences = []
#             lengths = []

#             for sat_id in sat_ids:
#                 seq_list = history_dict[sat_id]     # list of tensors of shape (feat_dim,)
#                 lengths.append(len(seq_list))

#                 # stack into (len_i, feat_dim)
#                 seq_tensor = torch.stack(seq_list, dim=0)
#                 sequences.append(seq_tensor)

#             # Pad to max length
#             max_len = max(lengths)
#             feat_dim = sequences[0].shape[-1]
#             num_sats = len(sequences)

#             sats_tensor = torch.zeros((num_sats, max_len, feat_dim), dtype=torch.float32, device=device)

#             for i, seq in enumerate(sequences):
#                 t = seq.shape[0]
#                 sats_tensor[i, :t] = seq.to(device)

#             lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

#             # ----- 3) ground truth -----
#             gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
            
#             pos_truth = lla_to_ecef(
#                 gt_row[['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].to_numpy().flatten()
#             )
#             vel_truth = heading_speed_to_ecef(
#                 gt_row['BearingDegrees'], gt_row['SpeedMps'],
#                 gt_row['LatitudeDegrees'], gt_row['LongitudeDegrees']
#             )

#             pos_truth = torch.tensor(pos_truth, dtype=torch.float32, device=device)
#             vel_truth = torch.tensor(vel_truth, dtype=torch.float32, device=device)

#             # ----- 4) forward pass through multi-epoch net -----
#             pr_weights, pr_error, prr_weights = net(sats_tensor, lengths_tensor)

#             pr_weight_reg = ((pr_weights.mean() - 0.5)**2) - 0.1*pr_weights.var()
#             prr_weight_reg = ((prr_weights.mean() - 0.5)**2) - 0.5*prr_weights.var()

#             Wx = torch.diag_embed(pr_weights.squeeze(1))
#             pr_error = pr_error.T.squeeze(0)

#             Wv = torch.diag_embed(prr_weights.squeeze(1))

#             # ----- 5) GNSS solver inputs for this epoch -----
#             pr = torch.tensor(epoch_df['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
#             prr = torch.tensor(epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
#             sat_pos = torch.tensor(epoch_df[['SvPositionXEcefMeters','SvPositionYEcefMeters','SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
#             sat_vel = torch.tensor(epoch_df[['SvVelocityXEcefMetersPerSecond','SvVelocityYEcefMetersPerSecond','SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)

#             # ----- baseline -----
#             pb = torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             prr_b = torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             pr_wb = torch.diag(pb)
#             prr_wb = torch.diag(prr_b)

#             baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_wb, Wv=prr_wb, prev_estimate=curr_pos)
#             pr_baseline_loss = torch.linalg.norm(baseline_pos['position'] - pos_truth)
#             prr_baseline_loss = torch.linalg.norm(baseline_pos['velocity'] - vel_truth)

#             # ----- corrected measurement -----
#             pr = pr + pr_error

#             curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=Wx, Wv=Wv, prev_estimate=curr_pos)

#             pr_loss = torch.linalg.norm(curr_pos['position'] - pos_truth)
#             prr_loss = torch.linalg.norm(curr_pos['velocity'] - vel_truth)

#             pos_gain = pr_baseline_loss.detach().item() - pr_loss.detach().item()
#             vel_gain = prr_baseline_loss.detach().item() - prr_loss.detach().item()

#             pos_loss = pr_loss / (pr_baseline_loss.detach().item() + 1e-6)
#             vel_loss = prr_loss / (prr_baseline_loss.detach().item() + 1e-6)

#             combined_loss = (
#                 pos_loss
#                 + 2.5 * vel_loss
#                 + 1 * torch.relu(pr_loss - pr_baseline_loss)
#                 + 2.5 * torch.relu(prr_loss - prr_baseline_loss)
#                 + 50 * pr_weight_reg**2 + 100 * prr_weight_reg**2
#             )

#             print(
#                 f"Epoch {epoch_id + 1} | "
#                 f"Pos err: {pr_loss.item():.3f} m (baseline {pr_baseline_loss.item():.3f}, Δ{pos_gain:.3f}) | "
#                 f"Vel err: {prr_loss.item():.3f} m/s (baseline {prr_baseline_loss.item():.3f}, Δ{vel_gain:.3f}) | "
#                 f"Combined Loss: {combined_loss.item():.3e}"
#             )

#             (combined_loss / 32).backward()
#             if epoch_id % 32 == 0 or epoch_id == df['epoch_id'].unique().max():
#                 optimizer.step()
#                 optimizer.zero_grad()

#             # detach the estimator state
#             curr_pos = {
#                 'position': curr_pos['position'].detach(),
#                 'velocity': curr_pos['velocity'].detach(),
#                 'clock_bias': curr_pos['clock_bias'].detach(),
#                 'clock_drift': curr_pos['clock_drift'].detach()
#             }

#     torch.save(net.state_dict(), "multi_epoch_network.pt")
#     print("Training complete, model saved")

epoch_manager = None

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats_np = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)

    # ----- 1) update temporal histories -----
    sat_ids = epoch_df["sat_identifier"].tolist()

    for sat_id, feat_vec in zip(sat_ids, feats):
        epoch_manager.add_entry(sat_id, feat_vec.detach().cpu())

    # ----- 2) retrieve per-satellite sequences -----
    history_dict = epoch_manager.batch_history(sat_ids)

    # Build padded tensor: (num_sats, max_len, feat_dim)
    sequences = []
    lengths = []

    for sat_id in sat_ids:
        seq_list = history_dict[sat_id]     # list of tensors of shape (feat_dim,)
        lengths.append(len(seq_list))

        # stack into (len_i, feat_dim)
        seq_tensor = torch.stack(seq_list, dim=0)
        sequences.append(seq_tensor)

    # Pad to max length
    max_len = max(lengths)
    feat_dim = sequences[0].shape[-1]
    num_sats = len(sequences)

    sats_tensor = torch.zeros((num_sats, max_len, feat_dim), dtype=torch.float32, device=device)

    for i, seq in enumerate(sequences):
        t = seq.shape[0]
        sats_tensor[i, :t] = seq.to(device)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return sats_tensor, lengths_tensor

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"
    data_iter = Data_file_iterator(base_path, split='train', limit=10)

    setup(data_iter, Gnss_multi_epoch_net)

    for df, truth_df in data_iter:
        print(f"Processing file: {data_iter.get_current_file_path()}")
        epoch_manager = Epoch_manager()
        run(df, truth_df, get_feats, save_path="multi_epoch_network.pt")