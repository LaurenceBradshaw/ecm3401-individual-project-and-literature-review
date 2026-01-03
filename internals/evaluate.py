import pandas as pd
import numpy as np
import torch
import folium
from internals.coord_systems import ecef_to_lla
from internals.gnss_positioning import Kalman_filter
import internals.gnss_positioning as gp
import internals.common as common

PR_COL = 'CorrectedPseudorange'
PRR_COL = 'CorrectedPseudorangeRateMetersPerSecond'
SAT_POS_COLS = ['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']
SAT_VEL_COLS = ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']

def setup(network_cls: torch.nn.Module, state_dict: str, kf: bool) -> None:
    global net, features, kf_enabled
    ###############
    kf_enabled = kf
    features = network_cls.features
    
    # Dummy mean and std since we are loading a pretrained model
    mean = torch.tensor([0 for _ in features], dtype=torch.float32).to(common.get_device())
    std = torch.tensor([1 for _ in features], dtype=torch.float32).to(common.get_device())

    net = network_cls(mean=mean,std=std,feat_dim=len(features))
    net.load_state_dict(torch.load(state_dict))
    net.eval()

# TODO: similar to train.py, create/use _compute_pos function to reduce code duplication
def run(df: pd.DataFrame, truth_df: pd.DataFrame, get_feats: callable, save_name: str) -> None:
    N = df['epoch_id'].nunique()
    truth_positions = np.zeros((N, 3))
    estimated_positions = np.zeros((N, 3))
    estimated_positions_baseline = np.zeros((N, 3))
    sat_trajectories = {}

    curr_pos = None
    kf = Kalman_filter()
    kf_baseline = Kalman_filter()
    # Initial position estimate using first epoch otherwise gradients are terrible initially
    pr = df[df['epoch_id'] == 0][PR_COL].to_numpy()
    prr = df[df['epoch_id'] == 0][PRR_COL].to_numpy()
    sat_pos = df[df['epoch_id'] == 0][SAT_POS_COLS].to_numpy()
    sat_vel = df[df['epoch_id'] == 0][SAT_VEL_COLS].to_numpy()

    curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)
    curr_pos["position"] = torch.tensor(curr_pos["position"], dtype=torch.float32)
    curr_pos["clock_bias"] = torch.tensor(curr_pos["clock_bias"], dtype=torch.float32)
    curr_pos["velocity"] = torch.tensor(curr_pos["velocity"], dtype=torch.float32)
    curr_pos["clock_drift"] = torch.tensor(curr_pos["clock_drift"], dtype=torch.float32)

    i = 0
    for epoch_id, epoch_df in df.groupby("epoch_id"):
        print(f"Processing epoch {epoch_id+1} / {N}")
        feats = epoch_df[features].to_numpy(dtype=np.float32)
        gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
        pos_truth = gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten()
        feats = torch.tensor(feats, dtype=torch.float32, device=common.get_device())
        pos_truth = torch.tensor(pos_truth, dtype=torch.float32, device=common.get_device())
        truth_positions[i, :] = pos_truth

        feats = get_feats(epoch_df, features, common.get_device())
        with torch.no_grad():
            pr_weights, pr_error, prr_weights = net(*feats)

        pr_weights = torch.diag_embed(pr_weights.squeeze(1))
        pr_error = pr_error.T.squeeze(0)

        prr_weights = torch.diag_embed(prr_weights.squeeze(1))

        pr = torch.tensor(epoch_df[PR_COL].to_numpy(), dtype=torch.float32, device='cpu')
        prr = torch.tensor(epoch_df[PRR_COL].to_numpy(), dtype=torch.float32, device='cpu')
        sat_pos = torch.tensor(epoch_df[SAT_POS_COLS].to_numpy(), dtype=torch.float32, device='cpu')
        sat_vel = torch.tensor(epoch_df[SAT_VEL_COLS].to_numpy(), dtype=torch.float32, device='cpu')

        pr_weights_baseline = torch.tensor(epoch_df['pr_baseline_weight'].to_numpy(), dtype=torch.float32, device='cpu')
        pr_weights_baseline = torch.diag(pr_weights_baseline)
        prr_weights_baseline = torch.tensor(epoch_df['prr_baseline_weight'].to_numpy(), dtype=torch.float32, device='cpu')
        prr_weights_baseline = torch.diag(prr_weights_baseline)
        baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights_baseline, Wv=prr_weights_baseline, prev_estimate=curr_pos)

        pr = pr + pr_error
        curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=pr_weights, Wv=prr_weights, prev_estimate=curr_pos)

        if kf_enabled:
            kf.predict()
            kf.update(curr_pos["position"].cpu().numpy(), curr_pos["velocity"].cpu().numpy(), curr_pos["clock_bias"].cpu().numpy(), curr_pos["clock_drift"].cpu().numpy())
            curr_pos["position"] = torch.tensor(kf.get_position(), dtype=torch.float32)
            curr_pos["velocity"] = torch.tensor(kf.get_velocity(), dtype=torch.float32)
            curr_pos["clock_bias"] = torch.tensor(kf.get_clock_bias(), dtype=torch.float32)
            curr_pos["clock_drift"] = torch.tensor(kf.get_clock_drift(), dtype=torch.float32)

            kf_baseline.predict()
            kf_baseline.update(baseline_pos["position"].cpu().numpy(), baseline_pos["velocity"].cpu().numpy(), baseline_pos["clock_bias"].cpu().numpy(), baseline_pos["clock_drift"].cpu().numpy())
            baseline_pos["position"] = torch.tensor(kf_baseline.get_position(), dtype=torch.float32)
            baseline_pos["velocity"] = torch.tensor(kf_baseline.get_velocity(), dtype=torch.float32)
            baseline_pos["clock_bias"] = torch.tensor(kf_baseline.get_clock_bias(), dtype=torch.float32)
            baseline_pos["clock_drift"] = torch.tensor(kf_baseline.get_clock_drift(), dtype=torch.float32)

        est_lla = ecef_to_lla(curr_pos["position"].cpu().numpy())
        est_lla_baseline = ecef_to_lla(baseline_pos["position"].cpu().numpy())
        estimated_positions[i, :] = est_lla
        estimated_positions_baseline[i, :] = est_lla_baseline

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
    
    m = folium.Map(
        location=[truth_positions[0,0], truth_positions[0,1]],
        zoom_start=15,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # Truth trajectory
    folium.PolyLine(
        list(zip(truth_positions[:,0], truth_positions[:,1])),
        color="red",
        weight=3,
        opacity=0.8,
        tooltip="Truth"
    ).add_to(m)

    # Estimated trajectory
    folium.PolyLine(
        list(zip(estimated_positions[:,0], estimated_positions[:,1])),
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Estimated"
    ).add_to(m)

    # Estimated trajectory unweighted
    folium.PolyLine(
        list(zip(estimated_positions_baseline[:,0], estimated_positions_baseline[:,1])),
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

    m.save(save_name)
    print(f"Saved {save_name}")