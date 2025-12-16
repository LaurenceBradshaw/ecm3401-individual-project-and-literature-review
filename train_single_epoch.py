import pandas as pd
import os
import numpy as np
from data_file_iter import Data_file_iterator
from model import Gnss_single_epoch_net
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.coord_systems import lla_to_ecef, ecef_to_lla, heading_speed_to_ecef
import utils.gnss_positioning as gp
from internals.train import setup, run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# if __name__ == "__main__":
#     base_path = "./smartphone-decimeter-2023/sdc2023"

#     data_iter = Data_file_iterator(base_path, split='train', limit=10)

#     features = Gnss_single_epoch_net.features

#     dataset_stats_file = f"{base_path}/dataset_stats.csv"
#     stats_df = pd.read_csv(dataset_stats_file)
#     mean = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'mean'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     std = torch.tensor([stats_df.loc[stats_df['column_name'] == col, 'std_dev'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     net = Gnss_single_epoch_net(
#         mean=mean,
#         std=std,
#         feat_dim=len(features)
#     ).to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#     for df in data_iter:
#         print(f"Processing file: {data_iter.get_current_file_path()}")
#         truth_df = data_iter.get_truth_file()

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
#         for epoch_id, epoch_df in df.groupby("epoch_id"):
#             if not epoch_df["poor_region"].all():
#                 continue
#             if epoch_id == 558:
#                 pass

#             feats = epoch_df[features].to_numpy(dtype=np.float32)
#             gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
#             pos_truth = lla_to_ecef(gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten())
#             vel_truth = heading_speed_to_ecef(gt_row['BearingDegrees'], gt_row['SpeedMps'], gt_row['LatitudeDegrees'], gt_row['LongitudeDegrees'])
#             feats = torch.tensor(feats, dtype=torch.float32, device=device)
#             pos_truth = torch.tensor(pos_truth, dtype=torch.float32, device=device)
#             vel_truth = torch.tensor(vel_truth, dtype=torch.float32, device=device)

#             # pr_weights, pr_error, prr_weights, prr_error = net(feats)
#             # pr_weights, prr_weights= net(feats)
#             pr_weights, pr_error, prr_weights= net(feats)

#             pr_weights = torch.diag_embed(pr_weights.squeeze(1))
#             pr_error = pr_error.T.squeeze(0)

#             prr_weights = torch.diag_embed(prr_weights.squeeze(1))
#             # prr_error = prr_error.T.squeeze(0)

#             pr = torch.tensor(epoch_df['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
#             prr = torch.tensor(epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
#             sat_pos = torch.tensor(epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
#             sat_vel = torch.tensor(epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)

#             pr_weights_baseline = torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             pr_weights_baseline = torch.diag(pr_weights_baseline)
#             prr_weights_baseline = torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             prr_weights_baseline = torch.diag(prr_weights_baseline)
#             baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights_baseline, Wv=prr_weights_baseline, prev_estimate=curr_pos)
#             pr_baseline_loss = torch.linalg.norm(baseline_pos['position'] - pos_truth)
#             prr_baseline_loss = torch.linalg.norm(baseline_pos['velocity'] - vel_truth)
#             # pr_baseline_loss = F.mse_loss(baseline_pos['position'], pos_truth)
#             # prr_baseline_loss = F.mse_loss(baseline_pos['velocity'], vel_truth)

#             pr = pr + pr_error
#             # prr = prr + prr_error
#             curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights, Wv=prr_weights, prev_estimate=curr_pos)
#             # pr_loss = F.mse_loss(curr_pos['position'], pos_truth)
#             # prr_loss = F.mse_loss(curr_pos['velocity'], vel_truth)
#             pr_loss = torch.linalg.norm(curr_pos['position'] - pos_truth)
#             prr_loss = torch.linalg.norm(curr_pos['velocity'] - vel_truth)


#             # # Ensure the predicted error corresponds to a physically consistant location
#             # clock_bias = curr_pos['clock_bias']  # scalar tensor
#             # x = torch.cat([curr_pos['position'], clock_bias.unsqueeze(0)], dim=0)  # (4,)
#             # pr_residuals = gp.pr_residuals_torch(x, sat_pos, pr)
#             # pr_res_loss = F.mse_loss(pr_error, pr_residuals)
#             # # Same for velocity
#             # clock_drift = curr_pos['clock_drift']
#             # v = torch.cat([curr_pos['velocity'], clock_drift.unsqueeze(0)], dim=0)
#             # prr_residuals = gp.prr_residuals_torch(v, sat_vel, prr, x, sat_pos)
#             # prr_res_loss = F.mse_loss(prr_error, prr_residuals)

#             # pos_rmse = torch.sqrt(pr_loss).item()
#             # pos_baseline_rmse = torch.sqrt(pr_baseline_loss).item()
#             # vel_rmse = torch.sqrt(prr_loss).item()
#             # vel_baseline_rmse = torch.sqrt(prr_baseline_loss).item()
#             # pos_gain = pos_baseline_rmse - pos_rmse
#             # vel_gain = vel_baseline_rmse - vel_rmse
#             pos_gain = pr_baseline_loss.detach().item() - pr_loss.detach().item()
#             vel_gain = prr_baseline_loss.detach().item() - prr_loss.detach().item()

#             pos_loss = pr_loss / (pr_baseline_loss.detach().item() + 1e-6)
#             vel_loss = prr_loss / (prr_baseline_loss.detach().item() + 1e-6)
#             combined_loss = (
#                 pos_loss
#                 + 2.5 * vel_loss
#                 + 1 * torch.relu(pr_loss - pr_baseline_loss)
#                 + 2.5 * torch.relu(prr_loss - prr_baseline_loss)
#                 # + 0.5 * pr_res_loss
#                 # + 1.25 * prr_res_loss
#             )

#             print(
#                 f"Epoch {epoch_id + 1} | "
#                 f"Pos RMSE: {pr_loss.item():.3f} m (baseline {pr_baseline_loss.item():.3f}, Δ{pos_gain:.3f}) | "
#                 f"Vel RMSE: {prr_loss.item():.3f} m/s (baseline {prr_baseline_loss.item():.3f}, Δ{vel_gain:.3f}) | "
#                 # f"Residual Consistency: pr {pr_res_loss.item():.3e}, prr {prr_res_loss.item():.3e} | "
#                 f"Combined Loss: {combined_loss.item():.3e}"
#             )

#             # multi = 2 if pr_baseline_loss.item() < pr_loss.item() else 1
#             # combined_loss += pr_loss * multi
#             eps = 1e-6
#             pr_entropy_reg = - (pr_weights * torch.log(pr_weights + eps) + (1-pr_weights) * torch.log(1-pr_weights + eps))
#             pr_entropy_loss = pr_entropy_reg.mean()
#             prr_entropy_reg = - (prr_weights * torch.log(prr_weights + eps) + (1-prr_weights) * torch.log(1-prr_weights + eps))
#             prr_entropy_loss = prr_entropy_reg.mean()
#             combined_loss = combined_loss + 0.5 * pr_entropy_loss + 0.1 * prr_entropy_loss
#             # error_mag_reg = torch.mean(error**2)
#             # loss = loss + 1e-3 * error_mag_reg

#             # torch.autograd.set_detect_anomaly(True)
#             (combined_loss/32).backward()
#             if epoch_id % 32 == 0 or epoch_id == df['epoch_id'].unique().max():
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             curr_pos = {
#                 'position': curr_pos['position'].detach(),
#                 'velocity': curr_pos['velocity'].detach(),
#                 'clock_bias': curr_pos['clock_bias'].detach(),
#                 'clock_drift': curr_pos['clock_drift'].detach()
#             }

#     torch.save(net.state_dict(), "single_epoch_network.pt")
#     print("Training complete, model saved")

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    feats = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
    return (feats,)

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"
    data_iter = Data_file_iterator(base_path, split='train', limit=10)

    setup(data_iter, Gnss_single_epoch_net)

    for df, truth_df in data_iter:
        print(f"Processing file: {data_iter.get_current_file_path()}")
        run(df, truth_df, get_feats, "single_epoch_network.pt")
