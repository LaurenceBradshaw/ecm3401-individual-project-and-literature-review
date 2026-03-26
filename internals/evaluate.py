import numpy as np
import matplotlib.pyplot as plt
import torch
import folium
import os
import pandas as pd
from pathlib import Path
from internals.drive_data import Drive
from internals.coord_systems import ecef_to_lla, errors_haversine, ecef_to_enu_rot
from internals.gnss_positioning import Kalman_filter
from internals.constants import PR_COL, PRR_COL, SAT_POS_COLS, SAT_VEL_COLS
import internals.gnss_positioning as gp
import internals.common as common

def path_to_name(path_str: str) -> str:
    path = Path(path_str)
    parts = path.parts

    if len(parts) < 2:
        raise ValueError("Path must contain at least two components")

    return f"{parts[-2]}_{parts[-1]}"

def compute_position_error_cdf(baseline_positions, nn_positions, truth_positions, num_points=1000):
    # Compute errors
    baseline_errors = np.linalg.norm(baseline_positions - truth_positions, axis=1)
    nn_errors = np.linalg.norm(nn_positions - truth_positions, axis=1)
    
    # Compute CDFs
    max_error = max(baseline_errors.max(), nn_errors.max())
    error_range = np.linspace(0, max_error, num_points)
    
    baseline_cdf = np.array([np.mean(baseline_errors <= e) for e in error_range])
    nn_cdf = np.array([np.mean(nn_errors <= e) for e in error_range])
    
    return {
        'error_range': error_range,
        'baseline_cdf': baseline_cdf,
        'nn_cdf': nn_cdf
    }

def print_global_error_all_files(results: list, save_dir: str, kf_string: str, save_postfix: str) -> None:
    total = {}

    for key in ["horizontal_sse", "vertical_sse", "3d_sse", "horizontal_baseline_sse", "vertical_baseline_sse", "3d_baseline_sse", 
                "velocity_horizontal_sse", "velocity_vertical_sse", "velocity_3d_sse", "velocity_horizontal_baseline_sse", "velocity_vertical_baseline_sse", "velocity_3d_baseline_sse", 
                "speed_sse", "speed_baseline_sse", "count", "speed_count"]:
        total[key] = sum(r[key] for r in results)

    horizontal_rmse = np.sqrt(total["horizontal_sse"] / total["count"])
    vertical_rmse = np.sqrt(total["vertical_sse"] / total["count"])
    rmse_3d = np.sqrt(total["3d_sse"] / total["count"])

    horizontal_baseline_rmse = np.sqrt(total["horizontal_baseline_sse"] / total["count"])
    vertical_baseline_rmse = np.sqrt(total["vertical_baseline_sse"] / total["count"])
    rmse_3d_baseline = np.sqrt(total["3d_baseline_sse"] / total["count"])

    horizontal_vel_rmse = np.sqrt(total["velocity_horizontal_sse"] / total["count"])
    vertical_vel_rmse = np.sqrt(total["velocity_vertical_sse"] / total["count"])
    vel_rmse_3d = np.sqrt(total["velocity_3d_sse"] / total["count"])

    horizontal_vel_baseline_rmse = np.sqrt(total["velocity_horizontal_baseline_sse"] / total["count"])
    vertical_vel_baseline_rmse = np.sqrt(total["velocity_vertical_baseline_sse"] / total["count"])
    vel_baseline_rmse_3d = np.sqrt(total["velocity_3d_baseline_sse"] / total["count"])

    speed_rmse = np.sqrt(total["speed_sse"] / total["speed_count"])
    speed_baseline_rmse = np.sqrt(total["speed_baseline_sse"] / total["speed_count"])

    all_pos_horizontal_errors = np.concatenate([r["pos_horizontal_errors"] for r in results])
    all_pos_vertical_errors = np.concatenate([r["pos_vertical_errors"] for r in results])
    all_pos_3d_errors = np.concatenate([r["pos_3d_errors"] for r in results])
    all_pos_horizontal_baseline_errors = np.concatenate([r["pos_horizontal_baseline_errors"] for r in results])
    all_pos_vertical_baseline_errors = np.concatenate([r["pos_vertical_baseline_errors"] for r in results])
    all_pos_3d_baseline_errors = np.concatenate([r["pos_3d_baseline_errors"] for r in results])
    all_vel_horizontal_errors = np.concatenate([r["vel_horizontal_errors"] for r in results])
    all_vel_vertical_errors = np.concatenate([r["vel_vertical_errors"] for r in results])
    all_vel_3d_errors = np.concatenate([r["vel_3d_errors"] for r in results])
    all_vel_horizontal_baseline_errors = np.concatenate([r["vel_horizontal_baseline_errors"] for r in results])
    all_vel_vertical_baseline_errors = np.concatenate([r["vel_vertical_baseline_errors"] for r in results])
    all_vel_3d_baseline_errors = np.concatenate([r["vel_3d_baseline_errors"] for r in results])
    all_speed_errors = np.concatenate([r["speed_errors"] for r in results])
    all_speed_baseline_errors = np.concatenate([r["speed_baseline_errors"] for r in results])

    pos_horizontal_p50 = np.percentile(all_pos_horizontal_errors, 50)
    pos_horizontal_p75 = np.percentile(all_pos_horizontal_errors, 75)
    pos_horizontal_p95 = np.percentile(all_pos_horizontal_errors, 95)
    pos_vertical_p50 = np.percentile(all_pos_vertical_errors, 50)
    pos_vertical_p75 = np.percentile(all_pos_vertical_errors, 75)
    pos_vertical_p95 = np.percentile(all_pos_vertical_errors, 95)
    pos_3d_p50 = np.percentile(all_pos_3d_errors, 50)
    pos_3d_p75 = np.percentile(all_pos_3d_errors, 75)
    pos_3d_p95 = np.percentile(all_pos_3d_errors, 95)
    pos_horizontal_baseline_p50 = np.percentile(all_pos_horizontal_baseline_errors, 50)
    pos_horizontal_baseline_p75 = np.percentile(all_pos_horizontal_baseline_errors, 75)
    pos_horizontal_baseline_p95 = np.percentile(all_pos_horizontal_baseline_errors, 95)
    pos_vertical_baseline_p50 = np.percentile(all_pos_vertical_baseline_errors, 50)
    pos_vertical_baseline_p75 = np.percentile(all_pos_vertical_baseline_errors, 75)
    pos_vertical_baseline_p95 = np.percentile(all_pos_vertical_baseline_errors, 95)
    pos_3d_baseline_p50 = np.percentile(all_pos_3d_baseline_errors, 50)
    pos_3d_baseline_p75 = np.percentile(all_pos_3d_baseline_errors, 75)
    pos_3d_baseline_p95 = np.percentile(all_pos_3d_baseline_errors, 95)
    vel_horizontal_p50 = np.percentile(all_vel_horizontal_errors, 50)
    vel_horizontal_p75 = np.percentile(all_vel_horizontal_errors, 75)
    vel_horizontal_p95 = np.percentile(all_vel_horizontal_errors, 95)
    vel_vertical_p50 = np.percentile(all_vel_vertical_errors, 50)
    vel_vertical_p75 = np.percentile(all_vel_vertical_errors, 75)
    vel_vertical_p95 = np.percentile(all_vel_vertical_errors, 95)
    vel_3d_p50 = np.percentile(all_vel_3d_errors, 50)
    vel_3d_p75 = np.percentile(all_vel_3d_errors, 75)
    vel_3d_p95 = np.percentile(all_vel_3d_errors, 95)
    vel_horizontal_baseline_p50 = np.percentile(all_vel_horizontal_baseline_errors, 50)
    vel_horizontal_baseline_p75 = np.percentile(all_vel_horizontal_baseline_errors, 75)
    vel_horizontal_baseline_p95 = np.percentile(all_vel_horizontal_baseline_errors, 95)
    vel_vertical_baseline_p50 = np.percentile(all_vel_vertical_baseline_errors, 50)
    vel_vertical_baseline_p75 = np.percentile(all_vel_vertical_baseline_errors, 75)
    vel_vertical_baseline_p95 = np.percentile(all_vel_vertical_baseline_errors, 95)
    vel_3d_baseline_p50 = np.percentile(all_vel_3d_baseline_errors, 50)
    vel_3d_baseline_p75 = np.percentile(all_vel_3d_baseline_errors, 75)
    vel_3d_baseline_p95 = np.percentile(all_vel_3d_baseline_errors, 95)
    speed_p50 = np.percentile(all_speed_errors, 50)
    speed_p75 = np.percentile(all_speed_errors, 75)
    speed_p95 = np.percentile(all_speed_errors, 95)
    speed_baseline_p50 = np.percentile(all_speed_baseline_errors, 50)
    speed_baseline_p75 = np.percentile(all_speed_baseline_errors, 75)
    speed_baseline_p95 = np.percentile(all_speed_baseline_errors, 95)

    print(
    f"\n=== GLOBAL RESULTS ACROSS {len(results)} FILES ==="
    )

    print(
        f"Position RMSE (horizontal): {horizontal_rmse:.2f} m "
        f"(baseline {horizontal_baseline_rmse:.2f} m, "
        f"improvement {(horizontal_baseline_rmse - horizontal_rmse)/horizontal_baseline_rmse*100:.2f} %)"
    )

    print(
        f"Position RMSE (vertical): {vertical_rmse:.2f} m "
        f"(baseline {vertical_baseline_rmse:.2f} m, "
        f"improvement {(vertical_baseline_rmse - vertical_rmse)/vertical_baseline_rmse*100:.2f} %)"
    )

    print(
        f"Position RMSE (3D): {rmse_3d:.2f} m "
        f"(baseline {rmse_3d_baseline:.2f} m, "
        f"improvement {(rmse_3d_baseline - rmse_3d)/rmse_3d_baseline*100:.2f} %)"
    )
    
    print(
        f"Velocity RMSE (horizontal): {horizontal_vel_rmse:.2f} m/s "
        f"(baseline {horizontal_vel_baseline_rmse:.2f} m/s, "
        f"improvement {(horizontal_vel_baseline_rmse - horizontal_vel_rmse)/horizontal_vel_baseline_rmse*100:.2f} %)"
    )

    print(
        f"Velocity RMSE (vertical): {vertical_vel_rmse:.2f} m/s "
        f"(baseline {vertical_vel_baseline_rmse:.2f} m/s, "
        f"improvement {(vertical_vel_baseline_rmse - vertical_vel_rmse)/vertical_vel_baseline_rmse*100:.2f} %)"
    )

    print(
        f"Velocity RMSE (3D): {vel_rmse_3d:.2f} m/s "
        f"(baseline {vel_baseline_rmse_3d:.2f} m/s, "
        f"improvement {(vel_baseline_rmse_3d - vel_rmse_3d)/vel_baseline_rmse_3d*100:.2f} %)"
    )

    print(
        f"Speed RMSE: {speed_rmse:.2f} m/s "
        f"(baseline {speed_baseline_rmse:.2f} m/s, "
        f"improvement {(speed_baseline_rmse - speed_rmse)/speed_baseline_rmse*100:.2f} %)"
    )

    print(
        f"Position P50/P75/P95 (horizontal): {pos_horizontal_p50:.2f} / {pos_horizontal_p75:.2f} / {pos_horizontal_p95:.2f} m "
        f"(baseline {pos_horizontal_baseline_p50:.2f} / {pos_horizontal_baseline_p75:.2f} / {pos_horizontal_baseline_p95:.2f} m, "
        f"improvement {(pos_horizontal_baseline_p95 - pos_horizontal_p95)/pos_horizontal_baseline_p95*100:.2f} % at P95)"
    )

    print(
        f"Position P50/P75/P95 (vertical): {pos_vertical_p50:.2f} / {pos_vertical_p75:.2f} / {pos_vertical_p95:.2f} m "
        f"(baseline {pos_vertical_baseline_p50:.2f} / {pos_vertical_baseline_p75:.2f} / {pos_vertical_baseline_p95:.2f} m, "
        f"improvement {(pos_vertical_baseline_p95 - pos_vertical_p95)/pos_vertical_baseline_p95*100:.2f} % at P95)"
    )

    print(
        f"Position P50/P75/P95 (3D): {pos_3d_p50:.2f} / {pos_3d_p75:.2f} / {pos_3d_p95:.2f} m "
        f"(baseline {pos_3d_baseline_p50:.2f} / {pos_3d_baseline_p75:.2f} / {pos_3d_baseline_p95:.2f} m, "
        f"improvement {(pos_3d_baseline_p95 - pos_3d_p95)/pos_3d_baseline_p95*100:.2f} % at P95)"
    )
    print(
        f"Velocity P50/P75/P95 (horizontal): {vel_horizontal_p50:.2f} / {vel_horizontal_p75:.2f} / {vel_horizontal_p95:.2f} m/s "
        f"(baseline {vel_horizontal_baseline_p50:.2f} / {vel_horizontal_baseline_p75:.2f} / {vel_horizontal_baseline_p95:.2f} m/s, "
        f"improvement {(vel_horizontal_baseline_p95 - vel_horizontal_p95)/vel_horizontal_baseline_p95*100:.2f} % at P95)"
    )

    print(
        f"Velocity P50/P75/P95 (vertical): {vel_vertical_p50:.2f} / {vel_vertical_p75:.2f} / {vel_vertical_p95:.2f} m/s "
        f"(baseline {vel_vertical_baseline_p50:.2f} / {vel_vertical_baseline_p75:.2f} / {vel_vertical_baseline_p95:.2f} m/s, "
        f"improvement {(vel_vertical_baseline_p95 - vel_vertical_p95)/vel_vertical_baseline_p95*100:.2f} % at P95)"
    )

    print(
        f"Velocity P50/P75/P95 (3D): {vel_3d_p50:.2f} / {vel_3d_p75:.2f} / {vel_3d_p95:.2f} m/s "
        f"(baseline {vel_3d_baseline_p50:.2f} / {vel_3d_baseline_p75:.2f} / {vel_3d_baseline_p95:.2f} m/s, "
        f"improvement {(vel_3d_baseline_p95 - vel_3d_p95)/vel_3d_baseline_p95*100:.2f} % at P95)"
    )

    print(
        f"Speed P50/P75/P95: {speed_p50:.2f} / {speed_p75:.2f} / {speed_p95:.2f} m/s "
        f"(baseline {speed_baseline_p50:.2f} / {speed_baseline_p75:.2f} / {speed_baseline_p95:.2f} m/s, "
        f"improvement {(speed_baseline_p95 - speed_p95)/speed_baseline_p95*100:.2f} % at P95)"
    )

    # Write global results to file
    file_name = os.path.join(save_dir, f"results_{kf_string}_{save_postfix}.txt")
    with open(file_name, "a") as f:
        f.write(f"=== Global results across {len(results)} files with {'KF' if kf_enabled else 'no KF'} ===\n")
        f.write(f"Position RMSE (horizontal): {horizontal_rmse:.2f} m (baseline {horizontal_baseline_rmse:.2f} m, improvement {(horizontal_baseline_rmse - horizontal_rmse)/horizontal_baseline_rmse*100:.2f} %)\n")
        f.write(f"Position RMSE (vertical):   {vertical_rmse:.2f} m (baseline {vertical_baseline_rmse:.2f} m, improvement {(vertical_baseline_rmse - vertical_rmse)/vertical_baseline_rmse*100:.2f} %)\n")
        f.write(f"Position RMSE (3D):         {rmse_3d:.2f} m (baseline {rmse_3d_baseline:.2f} m, improvement {(rmse_3d_baseline - rmse_3d)/rmse_3d_baseline*100:.2f} %)\n")
        f.write(f"Velocity RMSE (horizontal): {horizontal_vel_rmse:.2f} m/s (baseline {horizontal_vel_baseline_rmse:.2f} m/s, improvement {(horizontal_vel_baseline_rmse - horizontal_vel_rmse)/horizontal_vel_baseline_rmse*100:.2f} %)\n")
        f.write(f"Velocity RMSE (vertical):   {vertical_vel_rmse:.2f} m/s (baseline {vertical_vel_baseline_rmse:.2f} m/s, improvement {(vertical_vel_baseline_rmse - vertical_vel_rmse)/vertical_vel_baseline_rmse*100:.2f} %)\n")
        f.write(f"Velocity RMSE (3D):         {vel_rmse_3d:.2f} m/s (baseline {vel_baseline_rmse_3d:.2f} m/s, improvement {(vel_baseline_rmse_3d - vel_rmse_3d)/vel_baseline_rmse_3d*100:.2f} %)\n")
        f.write(f"Speed RMSE:                 {speed_rmse:.2f} m/s (baseline {speed_baseline_rmse:.2f} m/s, improvement {(speed_baseline_rmse - speed_rmse)/speed_baseline_rmse*100:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (horizontal): {pos_horizontal_p50:.2f} / {pos_horizontal_p75:.2f} / {pos_horizontal_p95:.2f} m (baseline {pos_horizontal_baseline_p50:.2f} / {pos_horizontal_baseline_p75:.2f} / {pos_horizontal_baseline_p95:.2f} m) (improvement {(pos_horizontal_baseline_p50 - pos_horizontal_p50)/pos_horizontal_baseline_p50*100:.2f} / {(pos_horizontal_baseline_p75 - pos_horizontal_p75)/pos_horizontal_baseline_p75*100:.2f} / {(pos_horizontal_baseline_p95 - pos_horizontal_p95)/pos_horizontal_baseline_p95*100:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (vertical):   {pos_vertical_p50:.2f} / {pos_vertical_p75:.2f} / {pos_vertical_p95:.2f} m (baseline {pos_vertical_baseline_p50:.2f} / {pos_vertical_baseline_p75:.2f} / {pos_vertical_baseline_p95:.2f} m) (improvement {(pos_vertical_baseline_p50 - pos_vertical_p50)/pos_vertical_baseline_p50*100:.2f} / {(pos_vertical_baseline_p75 - pos_vertical_p75)/pos_vertical_baseline_p75*100:.2f} / {(pos_vertical_baseline_p95 - pos_vertical_p95)/pos_vertical_baseline_p95*100:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (3D):         {pos_3d_p50:.2f} / {pos_3d_p75:.2f} / {pos_3d_p95:.2f} m (baseline {pos_3d_baseline_p50:.2f} / {pos_3d_baseline_p75:.2f} / {pos_3d_baseline_p95:.2f} m) (improvement {(pos_3d_baseline_p50 - pos_3d_p50)/pos_3d_baseline_p50*100:.2f} / {(pos_3d_baseline_p75 - pos_3d_p75)/pos_3d_baseline_p75*100:.2f} / {(pos_3d_baseline_p95 - pos_3d_p95)/pos_3d_baseline_p95*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (horizontal): {vel_horizontal_p50:.2f} / {vel_horizontal_p75:.2f} / {vel_horizontal_p95:.2f} m/s (baseline {vel_horizontal_baseline_p50:.2f} / {vel_horizontal_baseline_p75:.2f} / {vel_horizontal_baseline_p95:.2f} m/s) (improvement {(vel_horizontal_baseline_p50 - vel_horizontal_p50)/vel_horizontal_baseline_p50*100:.2f} / {(vel_horizontal_baseline_p75 - vel_horizontal_p75)/vel_horizontal_baseline_p75*100:.2f} / {(vel_horizontal_baseline_p95 - vel_horizontal_p95)/vel_horizontal_baseline_p95*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (vertical):   {vel_vertical_p50:.2f} / {vel_vertical_p75:.2f} / {vel_vertical_p95:.2f} m/s (baseline {vel_vertical_baseline_p50:.2f} / {vel_vertical_baseline_p75:.2f} / {vel_vertical_baseline_p95:.2f} m/s) (improvement {(vel_vertical_baseline_p50 - vel_vertical_p50)/vel_vertical_baseline_p50*100:.2f} / {(vel_vertical_baseline_p75 - vel_vertical_p75)/vel_vertical_baseline_p75*100:.2f} / {(vel_vertical_baseline_p95 - vel_vertical_p95)/vel_vertical_baseline_p95*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (3D):         {vel_3d_p50:.2f} / {vel_3d_p75:.2f} / {vel_3d_p95:.2f} m/s (baseline {vel_3d_baseline_p50:.2f} / {vel_3d_baseline_p75:.2f} / {vel_3d_baseline_p95:.2f} m/s) (improvement {(vel_3d_baseline_p50 - vel_3d_p50)/vel_3d_baseline_p50*100:.2f} / {(vel_3d_baseline_p75 - vel_3d_p75)/vel_3d_baseline_p75*100:.2f} / {(vel_3d_baseline_p95 - vel_3d_p95)/vel_3d_baseline_p95*100:.2f} %)\n")
        f.write(f"Speed P50/P75/P95:                 {speed_p50:.2f} / {speed_p75:.2f} / {speed_p95:.2f} m/s (baseline {speed_baseline_p50:.2f} / {speed_baseline_p75:.2f} / {speed_baseline_p95:.2f} m/s) (improvement {(speed_baseline_p50 - speed_p50)/speed_baseline_p50*100:.2f} / {(speed_baseline_p75 - speed_p75)/speed_baseline_p75*100:.2f} / {(speed_baseline_p95 - speed_p95)/speed_baseline_p95*100:.2f} %)\n")

def setup(network_cls: torch.nn.Module, state_dict: str, kf: bool) -> None:
    global net, features, kf_enabled
    ###############
    kf_enabled = kf
    features = network_cls.features
    
    # Dummy mean and std since we are loading a pretrained model
    mean = torch.tensor([0 for _ in features], dtype=torch.float64).to(common.get_device())
    std = torch.tensor([1 for _ in features], dtype=torch.float64).to(common.get_device())

    net = network_cls(mean=mean,std=std,feat_dim=len(features)).to(common.get_device())
    net.load_state_dict(torch.load(state_dict, map_location=common.get_device()))
    net.eval()

def run(drive: Drive, get_feats: callable, save_dir: str, save_postfix: str) -> None:
    df, truth_df = drive.get_dataframes()
    N = df['epoch_id'].nunique()
    truth_positions_lla = np.zeros((N, 3))
    truth_positions_ecef = np.zeros((N, 3))
    truth_velocity = np.zeros((N, 3))
    truth_speed = np.zeros((N,))
    estimated_positions_lla = np.zeros((N, 3))
    estimated_positions_ecef = np.zeros((N, 3))
    estimated_velocity = np.zeros((N, 3))
    estimated_speed = np.zeros((N,))
    estimated_positions_baseline_lla = np.zeros((N, 3))
    estimated_positions_baseline_ecef = np.zeros((N, 3))
    estimated_velocity_baseline = np.zeros((N, 3))
    estimated_speed_baseline = np.zeros((N,))
    pseudoranges = []
    sat_positions_ecef = []
    sat_trajectories = {}
    epoch_ids = []

    kf_string = "kf" if kf_enabled else "no_kf"

    curr_pos = None
    kf = Kalman_filter()
    kf_baseline = Kalman_filter()

    # Initial position estimate using first epoch otherwise gradients are terrible initially
    curr_pos = common.compute_pos(epoch_df=df[df['epoch_id'] == 0], pr_weights=None, pr_correction=None, prr_weights=None, prr_correction=None, curr_pos=curr_pos)
    curr_pos["position"] = torch.tensor(curr_pos["position"], dtype=torch.float64)
    curr_pos["clock_bias"] = torch.tensor(curr_pos["clock_bias"], dtype=torch.float64)
    curr_pos["velocity"] = torch.tensor(curr_pos["velocity"], dtype=torch.float64)
    curr_pos["clock_drift"] = torch.tensor(curr_pos["clock_drift"], dtype=torch.float64)

    baseline_pos = curr_pos.copy()

    prev_epoch_id = None
    i = 0
    for epoch_id, epoch_df in df.groupby("epoch_id"):
        print(f"Processing GNSS epoch {epoch_id+1} / {N}")
        epoch_ids.append(epoch_id)

        pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)
        # Remove clock features from truth. They're not useful here.
        pos_truth = pos_truth[:3]
        vel_truth = vel_truth[:3]
        
        # pos_truth = torch.tensor(pos_truth, dtype=torch.float64, device=common.get_device())
        truth_positions_lla[i, :] = ecef_to_lla(pos_truth.cpu().numpy())
        truth_positions_ecef[i, :] = pos_truth.cpu().numpy()
        truth_speed[i] = torch.linalg.norm(vel_truth).item() # TODO: Use vel_truth and split horizontal and vertical and 3d like with position.
        truth_velocity[i, :] = vel_truth.cpu().numpy()
        pseudoranges.append({row['sat_identifier']: row[PR_COL] for _, row in epoch_df.iterrows()})
        sat_positions_ecef.append({row['sat_identifier']: row[SAT_POS_COLS].to_numpy() for _, row in epoch_df.iterrows()})

        feats = get_feats(epoch_df, features, common.get_device())
        with torch.no_grad():
            Wx, pr_errors, Wv, prr_errors = net(*feats)

        Wx_baseline = torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float64, device=common.get_device())
        Wx_baseline = torch.diag(Wx_baseline)
        Wv_baseline = torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float64, device=common.get_device())
        Wv_baseline = torch.diag(Wv_baseline)

        # sigma_elev_pr = (3 / epoch_df['sin_elevation'])**2
        # sigma_cn0_pr = 5 * np.exp(-epoch_df['Cn0DbHz']/20)
        # pr_weights_baseline = 1 / (sigma_elev_pr + sigma_cn0_pr + 1e-6)
        # sigma_elev_prr = (0.1 / epoch_df['sin_elevation'])**2
        # sigma_cn0_prr = 0.05 * np.exp(-epoch_df['Cn0DbHz']/20)
        # prr_weights_baseline = 1 / (sigma_elev_prr + sigma_cn0_prr + 1e-6)
        # Wx_baseline = torch.diag(torch.tensor(pr_weights_baseline.to_numpy(), dtype=torch.float64, device=common.get_device()))
        # Wv_baseline = torch.diag(torch.tensor(prr_weights_baseline.to_numpy(), dtype=torch.float64, device=common.get_device()))

        baseline_pos = common.compute_pos_torch(epoch_df=epoch_df, pr_weights=Wx_baseline, pr_correction=None, prr_weights=Wv_baseline, prr_correction=None, curr_pos=baseline_pos)

        old_pos = curr_pos['position'].cpu().numpy()
        curr_pos = common.compute_pos_torch(epoch_df=epoch_df, pr_weights=Wx, pr_correction=pr_errors, prr_weights=Wv, prr_correction=prr_errors, curr_pos=curr_pos)

        if kf_enabled:
            if prev_epoch_id is None:
                dt = 1.0
            else:
                prev_epoch_df = df[df['epoch_id'] == prev_epoch_id]
                dt = (epoch_df['utcTimeMillis'].mean() - prev_epoch_df['utcTimeMillis'].mean()) / 1000.0
            sat_count = epoch_df.shape[0]
            model_speed = torch.linalg.norm(curr_pos["velocity"]).item()
            kf.predict(dt, model_speed)
            kf.update(
                curr_pos["position"].cpu().numpy(), 
                curr_pos["velocity"].cpu().numpy(), 
                curr_pos["clock_bias"].cpu().numpy(), 
                curr_pos["clock_drift"].cpu().numpy(),
                curr_pos['dop']['hdop'],
                curr_pos['dop']['vdop'],
                sat_count,
                model_speed
                )
            curr_pos["position"] = torch.tensor(kf.get_position(), dtype=torch.float64)
            curr_pos["velocity"] = torch.tensor(kf.get_velocity(), dtype=torch.float64)
            curr_pos["clock_bias"] = torch.tensor(kf.get_clock_bias(), dtype=torch.float64)
            curr_pos["clock_drift"] = torch.tensor(kf.get_clock_drift(), dtype=torch.float64)

            baseline_speed = torch.linalg.norm(baseline_pos["velocity"]).item()
            kf_baseline.predict(dt, baseline_speed)
            kf_baseline.update(baseline_pos["position"].cpu().numpy(), 
                               baseline_pos["velocity"].cpu().numpy(), 
                               baseline_pos["clock_bias"].cpu().numpy(), 
                               baseline_pos["clock_drift"].cpu().numpy(),
                               baseline_pos['dop']['hdop'],
                               baseline_pos['dop']['vdop'],
                               sat_count,
                               baseline_speed
                               )
            baseline_pos["position"] = torch.tensor(kf_baseline.get_position(), dtype=torch.float64)
            baseline_pos["velocity"] = torch.tensor(kf_baseline.get_velocity(), dtype=torch.float64)
            baseline_pos["clock_bias"] = torch.tensor(kf_baseline.get_clock_bias(), dtype=torch.float64)
            baseline_pos["clock_drift"] = torch.tensor(kf_baseline.get_clock_drift(), dtype=torch.float64)

        pos_diff = torch.linalg.norm(curr_pos["position"] - torch.tensor(old_pos, dtype=torch.float64))
        if pos_diff > 500: # if position jumps more than 500 m in one epoch, something is probably wrong - log for debugging
            pr = epoch_df[PR_COL].to_numpy()
            prr = epoch_df[PRR_COL].to_numpy()
            sat_pos = epoch_df[SAT_POS_COLS].to_numpy()
            sat_vel = epoch_df[SAT_VEL_COLS].to_numpy()
            print(f"Warning: Large position jump of {pos_diff:.2f} m at epoch {epoch_id}. Check satellite geometry and measurements.")
            print(f"Old position: {old_pos}, New position: {curr_pos['position'].cpu().numpy()}")
            print(f"Weights: {torch.diag(Wx).cpu().numpy()}\n {torch.diag(Wv).cpu().numpy()}")
            print(f"Errors: {pr_errors.cpu().numpy()}")
            print(f"Sat positions:\n{sat_pos}\nSat velocities:\n{sat_vel}\nPseudoranges:\n{(pr - pr_errors.cpu().numpy())}\nPRR:\n{prr}")

            prev_epoch_df = df[df['epoch_id'] == epoch_id - 1]
            prev_pr = prev_epoch_df[PR_COL].to_numpy()
            prev_prr = prev_epoch_df[PRR_COL].to_numpy()
            prev_sat_pos = prev_epoch_df[SAT_POS_COLS].to_numpy()
            prev_sat_vel = prev_epoch_df[SAT_VEL_COLS].to_numpy()
            print(f"Previous Sat positions:\n{prev_sat_pos}\nPrevious Sat velocities:\n{prev_sat_vel}\nPrevious Pseudoranges:\n{prev_pr}\nPrevious PRR:\n{prev_prr}")

            # write to file for further analysis
            debug_save_path = os.path.join(save_dir, f"debug_epoch_{epoch_id}_{path_to_name(drive.get_directory_name())}.txt")
            with open(debug_save_path, 'w') as f:
                f.write(f"Warning: Large position jump of {pos_diff:.2f} m at epoch {epoch_id}\n")
                f.write(f"Old position: {old_pos}, New position: {curr_pos['position'].cpu().numpy()}\n")
                f.write(f"Satellite order: {[row['sat_identifier'] for _, row in epoch_df.iterrows()]}\n")
                f.write(f"Weights: {torch.diag(Wx).cpu().numpy()}\n {torch.diag(Wv).cpu().numpy()}\n")
                f.write(f"Errors: {pr_errors.cpu().numpy()}\n")
                f.write(f"Sat positions:\n{sat_pos}\nSat velocities:\n{sat_vel}\nPseudoranges:\n{(pr - pr_errors.cpu().numpy())}\nPRR:\n{prr}\n")
                f.write(f"Previous Sat positions:\n{prev_sat_pos}\nPrevious Sat velocities:\n{prev_sat_vel}\nPrevious Pseudoranges:\n{prev_pr}\nPrevious PRR:\n{prev_prr}\n")
                f.write(f"Previous Satellite order: {[row['sat_identifier'] for _, row in prev_epoch_df.iterrows()]}\n")


        est_lla = ecef_to_lla(curr_pos["position"].cpu().numpy())
        est_lla_baseline = ecef_to_lla(baseline_pos["position"].cpu().numpy())
        estimated_positions_lla[i, :] = est_lla
        estimated_positions_ecef[i, :] = curr_pos["position"].cpu().numpy()
        estimated_speed[i] = np.linalg.norm(curr_pos["velocity"].cpu().numpy())
        estimated_velocity[i, :] = curr_pos["velocity"].cpu().numpy()
        estimated_positions_baseline_lla[i, :] = est_lla_baseline
        estimated_positions_baseline_ecef[i, :] = baseline_pos["position"].cpu().numpy()
        estimated_speed_baseline[i] = np.linalg.norm(baseline_pos["velocity"].cpu().numpy())
        estimated_velocity_baseline[i, :] = baseline_pos["velocity"].cpu().numpy()

        for _, row in epoch_df.iterrows():
            key = row['sat_identifier']

            if key not in sat_trajectories:
                sat_trajectories[key] = []

            sv_ecef = np.array([
                row["SvPositionXEcefMeters"],
                row["SvPositionYEcefMeters"],
                row["SvPositionZEcefMeters"],
            ])

            sv_lat, sv_lon, sv_alt = ecef_to_lla(sv_ecef)
            sat_trajectories[key].append((sv_lat, sv_lon))

        i += 1
        prev_epoch_id = epoch_id

    err_horizontal, err_vertical, err_3d = errors_haversine(estimated_positions_lla, truth_positions_lla)
    err_horizontal_baseline, err_vertical_baseline, err_3d_baseline = errors_haversine(estimated_positions_baseline_lla, truth_positions_lla)
    err_vertical = np.abs(err_vertical)
    err_vertical_baseline = np.abs(err_vertical_baseline)
    err_horizontal_rmse = np.sqrt(np.mean(err_horizontal**2))
    err_vertical_rmse = np.sqrt(np.mean(err_vertical**2))
    err_3d_rmse = np.sqrt(np.mean(err_3d**2))
    err_horizontal_baseline_rmse = np.sqrt(np.mean(err_horizontal_baseline**2))
    err_vertical_baseline_rmse = np.sqrt(np.mean(err_vertical_baseline**2))
    err_3d_baseline_rmse = np.sqrt(np.mean(err_3d_baseline**2))
    err_horizontal_improvement = (err_horizontal_baseline_rmse - err_horizontal_rmse) / err_horizontal_baseline_rmse * 100 if err_horizontal_baseline_rmse > 0 else 0
    err_vertical_improvement = (err_vertical_baseline_rmse - err_vertical_rmse) / err_vertical_baseline_rmse * 100 if err_vertical_baseline_rmse > 0 else 0
    err_3d_improvement = (err_3d_baseline_rmse - err_3d_rmse) / err_3d_baseline_rmse * 100 if err_3d_baseline_rmse > 0 else 0
    # Same for speed
    vel_error = estimated_velocity - truth_velocity
    vel_error_baseline = estimated_velocity_baseline - truth_velocity
    vel_error_enu = np.zeros_like(vel_error)
    vel_error_baseline_enu = np.zeros_like(vel_error)

    for i in range(len(vel_error)):
        ref_pos = truth_positions_ecef[i]

        R = ecef_to_enu_rot(ref_pos)

        vel_error_enu[i] = R @ vel_error[i]
        vel_error_baseline_enu[i] = R @ vel_error_baseline[i]

    vel_err_horizontal = np.sqrt(vel_error_enu[:,0]**2 + vel_error_enu[:,1]**2)
    vel_err_vertical = np.abs(vel_error_enu[:,2])
    vel_err_horizontal_baseline = np.sqrt(vel_error_baseline_enu[:,0]**2 + vel_error_baseline_enu[:,1]**2)
    vel_err_vertical_baseline = np.abs(vel_error_baseline_enu[:,2])
    vel_err_3d = np.linalg.norm(vel_error_enu, axis=1)
    vel_err_3d_baseline = np.linalg.norm(vel_error_baseline_enu, axis=1)

    vel_horizontal_rmse = np.sqrt(np.mean(vel_err_horizontal**2))
    vel_vertical_rmse = np.sqrt(np.mean(vel_err_vertical**2))
    vel_horizontal_baseline_rmse = np.sqrt(np.mean(vel_err_horizontal_baseline**2))
    vel_vertical_baseline_rmse = np.sqrt(np.mean(vel_err_vertical_baseline**2))
    vel_3d_rmse = np.sqrt(np.mean(vel_err_3d**2))
    vel_3d_baseline_rmse = np.sqrt(np.mean(vel_err_3d_baseline**2))
    vel_horizontal_improvement = (vel_horizontal_baseline_rmse - vel_horizontal_rmse) / vel_horizontal_baseline_rmse * 100 if vel_horizontal_baseline_rmse > 0 else 0
    vel_vertical_improvement = (vel_vertical_baseline_rmse - vel_vertical_rmse) / vel_vertical_baseline_rmse * 100 if vel_vertical_baseline_rmse > 0 else 0
    vel_3d_improvement = (vel_3d_baseline_rmse - vel_3d_rmse) / vel_3d_baseline_rmse * 100 if vel_3d_baseline_rmse > 0 else 0

    speed_diffs = estimated_speed - truth_speed
    speed_diffs = np.abs(speed_diffs)
    speed_diffs_baseline = estimated_speed_baseline - truth_speed
    speed_diffs_baseline = np.abs(speed_diffs_baseline)
    rmse_speed = np.sqrt(np.mean(speed_diffs**2))
    rmse_speed_baseline = np.sqrt(np.mean(speed_diffs_baseline**2))
    rmse_speed_improvement = (rmse_speed_baseline - rmse_speed) / rmse_speed_baseline * 100 if rmse_speed_baseline > 0 else 0
    
    print(f"Position RMSE (horizontal): {err_horizontal_rmse:.2f} m "
          f"(baseline {err_horizontal_baseline_rmse:.2f} m, "
          f"improvement {err_horizontal_improvement:.2f} %)"
          )
    print(f"Position RMSE (vertical): {err_vertical_rmse:.2f} m "
          f"(baseline {err_vertical_baseline_rmse:.2f} m, "
          f"improvement {err_vertical_improvement:.2f} %)"
          )
    print(f"Position RMSE (3D): {err_3d_rmse:.2f} m "
          f"(baseline {err_3d_baseline_rmse:.2f} m, "
          f"improvement {err_3d_improvement:.2f} %)"
          )
    print(f"Velocity RMSE (horizontal): {vel_horizontal_rmse:.2f} m/s "
          f"(baseline {vel_horizontal_baseline_rmse:.2f} m/s, "
          f"improvement {vel_horizontal_improvement:.2f} %)"
          )
    print(f"Velocity RMSE (vertical): {vel_vertical_rmse:.2f} m/s "
          f"(baseline {vel_vertical_baseline_rmse:.2f} m/s, "
          f"improvement {vel_vertical_improvement:.2f} %)"
          )
    print(f"Velocity RMSE (3D): {vel_3d_rmse:.2f} m/s "
          f"(baseline {vel_3d_baseline_rmse:.2f} m/s, "
          f"improvement {vel_3d_improvement:.2f} %)"
          )
    print(f"Speed RMSE: {rmse_speed:.2f} m/s "
          f"(baseline {rmse_speed_baseline:.2f} m/s, "
          f"improvement {rmse_speed_improvement:.2f} %)"
          )
    
    pos_horizontal_p50 = np.percentile(err_horizontal, 50)
    pos_horizontal_p75 = np.percentile(err_horizontal, 75)
    pos_horizontal_p95 = np.percentile(err_horizontal, 95)
    pos_vertical_p50 = np.percentile(err_vertical, 50)
    pos_vertical_p75 = np.percentile(err_vertical, 75)
    pos_vertical_p95 = np.percentile(err_vertical, 95)
    pos_3d_p50  = np.percentile(err_3d, 50)
    pos_3d_p75  = np.percentile(err_3d, 75)
    pos_3d_p95  = np.percentile(err_3d, 95)
    pos_horizontal_p50_baseline = np.percentile(err_horizontal_baseline, 50)
    pos_horizontal_p75_baseline = np.percentile(err_horizontal_baseline, 75)
    pos_horizontal_p95_baseline = np.percentile(err_horizontal_baseline, 95)
    pos_vertical_p50_baseline = np.percentile(err_vertical_baseline, 50)
    pos_vertical_p75_baseline = np.percentile(err_vertical_baseline, 75)
    pos_vertical_p95_baseline = np.percentile(err_vertical_baseline, 95)
    pos_3d_p50_baseline = np.percentile(err_3d_baseline, 50)
    pos_3d_p75_baseline = np.percentile(err_3d_baseline, 75)
    pos_3d_p95_baseline = np.percentile(err_3d_baseline, 95)

    vel_horizontal_p50 = np.percentile(vel_err_horizontal, 50)
    vel_horizontal_p75 = np.percentile(vel_err_horizontal, 75)
    vel_horizontal_p95 = np.percentile(vel_err_horizontal, 95)
    vel_vertical_p50 = np.percentile(vel_err_vertical, 50)
    vel_vertical_p75 = np.percentile(vel_err_vertical, 75)
    vel_vertical_p95 = np.percentile(vel_err_vertical, 95)
    vel_3d_p50  = np.percentile(vel_err_3d, 50)
    vel_3d_p75  = np.percentile(vel_err_3d, 75)
    vel_3d_p95  = np.percentile(vel_err_3d, 95)
    vel_horizontal_p50_baseline = np.percentile(vel_err_horizontal_baseline, 50)
    vel_horizontal_p75_baseline = np.percentile(vel_err_horizontal_baseline, 75)
    vel_horizontal_p95_baseline = np.percentile(vel_err_horizontal_baseline, 95)
    vel_vertical_p50_baseline = np.percentile(vel_err_vertical_baseline, 50)
    vel_vertical_p75_baseline = np.percentile(vel_err_vertical_baseline, 75)
    vel_vertical_p95_baseline = np.percentile(vel_err_vertical_baseline, 95)
    vel_3d_p50_baseline = np.percentile(vel_err_3d_baseline, 50)
    vel_3d_p75_baseline = np.percentile(vel_err_3d_baseline, 75)
    vel_3d_p95_baseline = np.percentile(vel_err_3d_baseline, 95)

    speed_p50 = np.percentile(speed_diffs, 50)
    speed_p75 = np.percentile(speed_diffs, 75)
    speed_p95 = np.percentile(speed_diffs, 95)
    speed_p50_baseline = np.percentile(speed_diffs_baseline, 50)
    speed_p75_baseline = np.percentile(speed_diffs_baseline, 75)
    speed_p95_baseline = np.percentile(speed_diffs_baseline, 95)

    # Write rmse results to file
    results_save_name = os.path.join(save_dir, f"results_{kf_string}_{save_postfix}.txt")
    with open(results_save_name, "a") as f:
        f.write(f"Results for drive {drive.get_directory_name()} with {'KF' if kf_enabled else 'no KF'}\n")
        f.write(f"Position RMSE (horizontal): {err_horizontal_rmse:.2f} m (baseline {err_horizontal_baseline_rmse:.2f} m, improvement {err_horizontal_improvement:.2f} %)\n")
        f.write(f"Position RMSE (vertical):   {err_vertical_rmse:.2f} m (baseline {err_vertical_baseline_rmse:.2f} m, improvement {err_vertical_improvement:.2f} %)\n")
        f.write(f"Position RMSE (3D):         {err_3d_rmse:.2f} m (baseline {err_3d_baseline_rmse:.2f} m, improvement {err_3d_improvement:.2f} %)\n")
        f.write(f"Velocity RMSE (horizontal): {vel_horizontal_rmse:.2f} m/s (baseline {vel_horizontal_baseline_rmse:.2f} m/s, improvement {vel_horizontal_improvement:.2f} %)\n")
        f.write(f"Velocity RMSE (vertical):   {vel_vertical_rmse:.2f} m/s (baseline {vel_vertical_baseline_rmse:.2f} m/s, improvement {vel_vertical_improvement:.2f} %)\n")
        f.write(f"Velocity RMSE (3D):         {vel_3d_rmse:.2f} m/s (baseline {vel_3d_baseline_rmse:.2f} m/s, improvement {vel_3d_improvement:.2f} %)\n")
        f.write(f"Speed RMSE:                 {rmse_speed:.2f} m/s (baseline {rmse_speed_baseline:.2f} m/s, improvement {rmse_speed_improvement:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (horizontal): {pos_horizontal_p50:.2f} / {pos_horizontal_p75:.2f} / {pos_horizontal_p95:.2f} m (baseline {pos_horizontal_p50_baseline:.2f} / {pos_horizontal_p75_baseline:.2f} / {pos_horizontal_p95_baseline:.2f} m, improvement {(pos_horizontal_p50_baseline - pos_horizontal_p50)/pos_horizontal_p50_baseline*100:.2f} / {(pos_horizontal_p75_baseline - pos_horizontal_p75)/pos_horizontal_p75_baseline*100:.2f} / {(pos_horizontal_p95_baseline - pos_horizontal_p95)/pos_horizontal_p95_baseline*100:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (vertical):   {pos_vertical_p50:.2f} / {pos_vertical_p75:.2f} / {pos_vertical_p95:.2f} m (baseline {pos_vertical_p50_baseline:.2f} / {pos_vertical_p75_baseline:.2f} / {pos_vertical_p95_baseline:.2f} m, improvement {(pos_vertical_p50_baseline - pos_vertical_p50)/pos_vertical_p50_baseline*100:.2f} / {(pos_vertical_p75_baseline - pos_vertical_p75)/pos_vertical_p75_baseline*100:.2f} / {(pos_vertical_p95_baseline - pos_vertical_p95)/pos_vertical_p95_baseline*100:.2f} %)\n")
        f.write(f"Position P50/P75/P95 (3D):         {pos_3d_p50:.2f} / {pos_3d_p75:.2f} / {pos_3d_p95:.2f} m (baseline {pos_3d_p50_baseline:.2f} / {pos_3d_p75_baseline:.2f} / {pos_3d_p95_baseline:.2f} m, improvement {(pos_3d_p50_baseline - pos_3d_p50)/pos_3d_p50_baseline*100:.2f} / {(pos_3d_p75_baseline - pos_3d_p75)/pos_3d_p75_baseline*100:.2f} / {(pos_3d_p95_baseline - pos_3d_p95)/pos_3d_p95_baseline*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (horizontal): {vel_horizontal_p50:.2f} / {vel_horizontal_p75:.2f} / {vel_horizontal_p95:.2f} m/s (baseline {vel_horizontal_p50_baseline:.2f} / {vel_horizontal_p75_baseline:.2f} / {vel_horizontal_p95_baseline:.2f} m/s, improvement {(vel_horizontal_p50_baseline - vel_horizontal_p50)/vel_horizontal_p50_baseline*100:.2f} / {(vel_horizontal_p75_baseline - vel_horizontal_p75)/vel_horizontal_p75_baseline*100:.2f} / {(vel_horizontal_p95_baseline - vel_horizontal_p95)/vel_horizontal_p95_baseline*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (vertical):   {vel_vertical_p50:.2f} / {vel_vertical_p75:.2f} / {vel_vertical_p95:.2f} m/s (baseline {vel_vertical_p50_baseline:.2f} / {vel_vertical_p75_baseline:.2f} / {vel_vertical_p95_baseline:.2f} m/s, improvement {(vel_vertical_p50_baseline - vel_vertical_p50)/vel_vertical_p50_baseline*100:.2f} / {(vel_vertical_p75_baseline - vel_vertical_p75)/vel_vertical_p75_baseline*100:.2f} / {(vel_vertical_p95_baseline - vel_vertical_p95)/vel_vertical_p95_baseline*100:.2f} %)\n")
        f.write(f"Velocity P50/P75/P95 (3D):         {vel_3d_p50:.2f} / {vel_3d_p75:.2f} / {vel_3d_p95:.2f} m/s (baseline {vel_3d_p50_baseline:.2f} / {vel_3d_p75_baseline:.2f} / {vel_3d_p95_baseline:.2f} m/s, improvement {(vel_3d_p50_baseline - vel_3d_p50)/vel_3d_p50_baseline*100:.2f} / {(vel_3d_p75_baseline - vel_3d_p75)/vel_3d_p75_baseline*100:.2f} / {(vel_3d_p95_baseline - vel_3d_p95)/vel_3d_p95_baseline*100:.2f} %)\n")
        f.write(f"Speed P50/P75/P95:                 {speed_p50:.2f} / {speed_p75:.2f} / {speed_p95:.2f} m/s (baseline {speed_p50_baseline:.2f} / {speed_p75_baseline:.2f} / {speed_p95_baseline:.2f} m/s, improvement {(speed_p50_baseline - speed_p50)/speed_p50_baseline*100:.2f} / {(speed_p75_baseline - speed_p75)/speed_p75_baseline*100:.2f} / {(speed_p95_baseline - speed_p95)/speed_p95_baseline*100:.2f} %)\n")

    m = folium.Map(
        location=[truth_positions_lla[0,0], truth_positions_lla[0,1]],
        zoom_start=15,
        max_zoom=21,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # Truth trajectory
    folium.PolyLine(
        list(zip(truth_positions_lla[:,0], truth_positions_lla[:,1])),
        color="red",
        weight=3,
        opacity=0.8,
        tooltip="Truth"
    ).add_to(m)

    # Estimated trajectory
    folium.PolyLine(
        list(zip(estimated_positions_lla[:,0], estimated_positions_lla[:,1])),
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Estimated"
    ).add_to(m)

    # Estimated trajectory unweighted
    folium.PolyLine(
        list(zip(estimated_positions_baseline_lla[:,0], estimated_positions_baseline_lla[:,1])),
        color="green",
        weight=3,
        opacity=0.8,
        tooltip="Estimated Baseline"
    ).add_to(m)

    for sv_id, path in sat_trajectories.items():
        if len(path) < 2:
            continue

        # satellite paths often span sky: plot lightly
        folium.PolyLine(
            path,
            color="purple",
            weight=1,
            opacity=0.5,
            tooltip=f"SV {sv_id}"
        ).add_to(m)

    # Add satellite markers at first/last point
    for sv_id, path in sat_trajectories.items():
        if len(path) < 2:
            continue
        folium.CircleMarker(location=path[0], radius=3, color="purple", fill=True).add_to(m)
        folium.CircleMarker(location=path[-1], radius=3, color="black", fill=True).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    map_save_name = os.path.join(save_dir, path_to_name(drive.get_directory_name()) + f"_trajectory_map_{kf_string}_{save_postfix}.html")
    m.save(map_save_name)
    print(f"Saved {map_save_name}")

    cdf_save_name = os.path.join(save_dir, path_to_name(drive.get_directory_name()) + f"_error_cdf_{kf_string}_{save_postfix}.png")
    cdf_result = compute_position_error_cdf(estimated_positions_baseline_ecef, estimated_positions_ecef, truth_positions_ecef)
    # plt.figure(figsize=(12, 8))
    plt.figure(figsize=(8, 5))
    plt.plot(cdf_result['error_range'], cdf_result['baseline_cdf'], label='Baseline')
    plt.plot(cdf_result['error_range'], cdf_result['nn_cdf'], label='NN Estimate')
    plt.xlabel('Position Error (m)')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(cdf_save_name, dpi=200, bbox_inches='tight')
    plt.close()

    # plot errors horizontal/vertical/3d over time
    err_over_time_save_name = os.path.join(save_dir, path_to_name(drive.get_directory_name()) + f"_error_over_time_{kf_string}_{save_postfix}.png")
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(err_horizontal, label="Horizontal Error (m)")
    plt.plot(err_horizontal_baseline, label="Horizontal Error Baseline (m)", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Error (m)")
    plt.title("Horizontal Position Error Over Time")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(err_vertical, label="Vertical Error (m)", color='orange')
    plt.plot(err_vertical_baseline, label="Vertical Error Baseline (m)", linestyle='--', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Error (m)")
    plt.title("Vertical Position Error Over Time")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(err_3d, label="3D Error (m)", color='green')
    plt.plot(err_3d_baseline, label="3D Error Baseline (m)", linestyle='--', color='darkgreen')
    plt.xlabel("Epoch")
    plt.ylabel("Error (m)")
    plt.title("3D Position Error Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(err_over_time_save_name)
    plt.close()

    # Write trajectory csv
    trajectory_save_name = os.path.join(save_dir, path_to_name(drive.get_directory_name()) + f"_trajectory_{kf_string}_{save_postfix}.csv")
    trajectory_df = pd.DataFrame({
        "truth_lat": truth_positions_lla[:,0],
        "truth_lon": truth_positions_lla[:,1],
        # "truth_alt": truth_positions_lla[:,2],
        "est_lat": estimated_positions_lla[:,0],
        "est_lon": estimated_positions_lla[:,1],
        # "est_alt": estimated_positions_lla[:,2],
        "est_baseline_lat": estimated_positions_baseline_lla[:,0],
        "est_baseline_lon": estimated_positions_baseline_lla[:,1],
        # "est_baseline_alt": estimated_positions_baseline_lla[:,2],
        # "truth_vel_x": truth_velocity[:,0],
        # "truth_vel_y": truth_velocity[:,1],
        # "truth_vel_z": truth_velocity[:,2],
        # "est_vel_x": estimated_velocity[:,0],
        # "est_vel_y": estimated_velocity[:,1],
        # "est_vel_z": estimated_velocity[:,2],
        # "est_baseline_vel_x": estimated_velocity_baseline[:,0],
        # "est_baseline_vel_y": estimated_velocity_baseline[:,1],
        # "est_baseline_vel_z": estimated_velocity_baseline[:,2],
        # "truth_speed": truth_speed,
        # "est_speed": estimated_speed,
        # "est_baseline_speed": estimated_speed_baseline,
        "epoch_id": epoch_ids
    })
    trajectory_df.to_csv(trajectory_save_name, index=False)


    return {
        "horizontal_sse": float(np.sum(err_horizontal**2)),
        "vertical_sse": float(np.sum(err_vertical**2)),
        "3d_sse": float(np.sum(err_3d**2)),
        "horizontal_baseline_sse": float(np.sum(err_horizontal_baseline**2)),
        "vertical_baseline_sse": float(np.sum(err_vertical_baseline**2)),
        "3d_baseline_sse": float(np.sum(err_3d_baseline**2)),
        "velocity_horizontal_sse": float(np.sum(vel_err_horizontal**2)),
        "velocity_vertical_sse": float(np.sum(vel_err_vertical**2)),
        "velocity_3d_sse": float(np.sum(vel_err_3d**2)),
        "velocity_horizontal_baseline_sse": float(np.sum(vel_err_horizontal_baseline**2)),
        "velocity_vertical_baseline_sse": float(np.sum(vel_err_vertical_baseline**2)),
        "velocity_3d_baseline_sse": float(np.sum(vel_err_3d_baseline**2)),
        "speed_sse": float(np.sum(speed_diffs**2)),
        "speed_baseline_sse": float(np.sum(speed_diffs_baseline**2)),
        "count": int(len(err_horizontal)),
        "speed_count": int(len(speed_diffs)),
        "pos_horizontal_errors": err_horizontal.tolist(),
        "pos_vertical_errors": err_vertical.tolist(),
        "pos_3d_errors": err_3d.tolist(),
        "vel_horizontal_errors": vel_err_horizontal.tolist(),
        "vel_vertical_errors": vel_err_vertical.tolist(),
        "vel_3d_errors": vel_err_3d.tolist(),
        "speed_errors": speed_diffs.tolist(),
        "pos_horizontal_baseline_errors": err_horizontal_baseline.tolist(),
        "pos_vertical_baseline_errors": err_vertical_baseline.tolist(),
        "pos_3d_baseline_errors": err_3d_baseline.tolist(),
        "vel_horizontal_baseline_errors": vel_err_horizontal_baseline.tolist(),
        "vel_vertical_baseline_errors": vel_err_vertical_baseline.tolist(),
        "vel_3d_baseline_errors": vel_err_3d_baseline.tolist(),
        "speed_baseline_errors": speed_diffs_baseline.tolist(),
    }