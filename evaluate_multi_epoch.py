import pandas as pd
import os
import numpy as np
from data_file_iter import Data_file_iterator
from model import Gnss_multi_epoch_net
import torch
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
# from utils.coord_systems import lla_to_ecef, ecef_to_lla
# import utils.gnss_positioning as gp
# import folium
# from utils.gnss_positioning import Kalman_filter
from epoch_manager import Epoch_manager
from internals.evaluate import setup, run


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_device(device)

# if __name__ == "__main__":
#     base_path = "./smartphone-decimeter-2023/sdc2023"

#     features = Gnss_multi_epoch_net.features

#     dataset_stats_file = f"{base_path}/dataset_stats.csv"
#     stats_df = pd.read_csv(dataset_stats_file)
#     mean = torch.tensor([
#         stats_df.loc[stats_df['column_name'] == col, 'mean'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     std = torch.tensor([
#         stats_df.loc[stats_df['column_name'] == col, 'std_dev'].values[0] for col in features
#     ], dtype=torch.float32).to(device)

#     net = Gnss_multi_epoch_net(mean=mean,std=std,feat_dim=len(features))
#     net.load_state_dict(torch.load("multi_epoch_network.pt"))
#     net.eval()

#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
#     data_iter = Data_file_iterator(base_path, split='train', prefix='2021-07-14-20-50-us-ca-mtv-e')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-04-01-18-22-us-ca-lax-t')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-09-06-00-01-us-ca-routen')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-03-08-21-34-us-ca-mtv-u')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-05-09-21-32-us-ca-mtv-pe1')
#     for df, truth_df in data_iter:
#         print(f"Evaluating file: {data_iter.get_current_file_path()}")
#         N = df['epoch_id'].nunique()
#         truth_positions = np.zeros((N, 3))
#         estimated_positions = np.zeros((N, 3))
#         estimated_positions_baseline = np.zeros((N, 3))
#         sat_trajectories = {}

#         curr_pos = None
#         kf = Kalman_filter()
#         kf_baseline = Kalman_filter()
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

#         i = 0
#         epoch_manager = Epoch_manager()
#         for epoch_id, epoch_df in df.groupby("epoch_id"):
#             print(f"Processing epoch {epoch_id+1} / {N}")
#             feats = epoch_df[features].to_numpy(dtype=np.float32)
#             gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
#             pos_truth = gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten()
#             feats = torch.tensor(feats, dtype=torch.float32, device=device)
#             pos_truth = torch.tensor(pos_truth, dtype=torch.float32, device=device)
#             truth_positions[i, :] = pos_truth

#             sat_ids = epoch_df["sat_identifier"].tolist()

#             for sat_id, feat_vec in zip(sat_ids, feats):
#                 epoch_manager.add_entry(sat_id, feat_vec.detach().cpu())

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
#             for j, seq in enumerate(sequences):
#                 t = seq.shape[0]
#                 sats_tensor[j, :t] = seq.to(device)

#             lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

#             with torch.no_grad():
#                 pr_weights, pr_error, prr_weights = net(sats_tensor, lengths_tensor)

#             pr_weights = torch.diag_embed(pr_weights.squeeze(1))
#             pr_error = pr_error.T.squeeze(0)

#             prr_weights = torch.diag_embed(prr_weights.squeeze(1))

#             pr = torch.tensor(epoch_df['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
#             prr = torch.tensor(epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
#             sat_pos = torch.tensor(epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
#             sat_vel = torch.tensor(epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)

#             pr_weights_baseline = torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             pr_weights_baseline = torch.diag(pr_weights_baseline)
#             prr_weights_baseline = torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float32)
#             prr_weights_baseline = torch.diag(prr_weights_baseline)
#             baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights_baseline, Wv=prr_weights_baseline, prev_estimate=curr_pos)
#             # baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=baseline_weights_diag, prev_estimate=curr_pos)

#             pr = pr + pr_error
#             curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights, Wv=prr_weights, prev_estimate=curr_pos)

#             # kf.predict()
#             # kf.update(curr_pos["position"].cpu().numpy(), curr_pos["velocity"].cpu().numpy(), curr_pos["clock_bias"].cpu().numpy(), curr_pos["clock_drift"].cpu().numpy())
#             # curr_pos["position"] = torch.tensor(kf.get_position(), dtype=torch.float32)
#             # curr_pos["velocity"] = torch.tensor(kf.get_velocity(), dtype=torch.float32)
#             # curr_pos["clock_bias"] = torch.tensor(kf.get_clock_bias(), dtype=torch.float32)
#             # curr_pos["clock_drift"] = torch.tensor(kf.get_clock_drift(), dtype=torch.float32)

#             # kf_baseline.predict()
#             # kf_baseline.update(baseline_pos["position"].cpu().numpy(), baseline_pos["velocity"].cpu().numpy(), baseline_pos["clock_bias"].cpu().numpy(), baseline_pos["clock_drift"].cpu().numpy())
#             # baseline_pos["position"] = torch.tensor(kf_baseline.get_position(), dtype=torch.float32)
#             # baseline_pos["velocity"] = torch.tensor(kf_baseline.get_velocity(), dtype=torch.float32)
#             # baseline_pos["clock_bias"] = torch.tensor(kf_baseline.get_clock_bias(), dtype=torch.float32)
#             # baseline_pos["clock_drift"] = torch.tensor(kf_baseline.get_clock_drift(), dtype=torch.float32)

#             est_lla = ecef_to_lla(curr_pos["position"].cpu().numpy())
#             est_lla_baseline = ecef_to_lla(baseline_pos["position"].cpu().numpy())
#             estimated_positions[i, :] = est_lla
#             estimated_positions_baseline[i, :] = est_lla_baseline

#             for _, row in epoch_df.iterrows():
#                 # sv_id = str(row["ConstellationType"]) + '_' + str(row["Svid"])  # or row["ConstellationType"] + row["Svid"] if mixing GNSS constellations
#                 key = row['sat_identifier']

#                 if key not in sat_trajectories:
#                     sat_trajectories[key] = []

#                 sv_ecef = np.array([
#                     row["SvPositionXEcefMeters"],
#                     row["SvPositionYEcefMeters"],
#                     row["SvPositionZEcefMeters"],
#                 ])

#                 sv_lat, sv_lon, sv_alt = ecef_to_lla(sv_ecef)
#                 sat_trajectories[key].append((sv_lat, sv_lon))

#             i += 1
        
#         m = folium.Map(
#             location=[truth_positions[0,0], truth_positions[0,1]],
#             zoom_start=15,
#             tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
#             attr="Google"
#         )

#         # Truth trajectory
#         folium.PolyLine(
#             list(zip(truth_positions[:,0], truth_positions[:,1])),
#             color="red",
#             weight=3,
#             opacity=0.8,
#             tooltip="Truth"
#         ).add_to(m)

#         # Estimated trajectory
#         folium.PolyLine(
#             list(zip(estimated_positions[:,0], estimated_positions[:,1])),
#             color="blue",
#             weight=3,
#             opacity=0.8,
#             tooltip="Estimated"
#         ).add_to(m)

#         # Estimated trajectory unweighted
#         folium.PolyLine(
#             list(zip(estimated_positions_baseline[:,0], estimated_positions_baseline[:,1])),
#             color="green",
#             weight=3,
#             opacity=0.8,
#             tooltip="Estimated Baseline"
#         ).add_to(m)

#         for sv_id, path in sat_trajectories.items():
#             if len(path) < 2:
#                 continue

#             # satellite paths often span sky: plot lightly
#             folium.PolyLine(
#                 path,
#                 color="purple",
#                 weight=1,
#                 opacity=0.5,
#                 tooltip=f"SV {sv_id}"
#             ).add_to(m)

#         # Add satellite markers at first/last point
#         for sv_id, path in sat_trajectories.items():
#             if len(path) < 2:
#                 continue
#             folium.CircleMarker(location=path[0], radius=3, color="purple", fill=True).add_to(m)
#             folium.CircleMarker(location=path[-1], radius=3, color="black", fill=True).add_to(m)

#         m.save("trajectory_map_multi_epoch.html")
#         print("Saved trajectory_map_multi_epoch.html")
#         break

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

    setup(Gnss_multi_epoch_net, "multi_epoch_network.pt", kf=False)
    for df, truth_df in data_iter:
        print(f"Evaluating file: {data_iter.get_current_file_path()}")
        save_name = os.path.basename(os.path.dirname(os.path.dirname(data_iter.get_current_file_path()))) + "_trajectory_map_multi_epoch.html"
        epoch_manager = Epoch_manager()
        run(df, truth_df, get_feats, save_name)
        break

    
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
#     data_iter = Data_file_iterator(base_path, split='train', prefix='2021-07-14-20-50-us-ca-mtv-e')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-04-01-18-22-us-ca-lax-t')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-09-06-00-01-us-ca-routen')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-03-08-21-34-us-ca-mtv-u')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-05-09-21-32-us-ca-mtv-pe1')