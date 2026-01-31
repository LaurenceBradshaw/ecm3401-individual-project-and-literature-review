import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import folium
from folium.plugins import TimestampedGeoJson
from internals.coord_systems import ecef_to_lla, lla_to_ecef
from internals.gnss_positioning import Kalman_filter
from internals.constants import PR_COL, PRR_COL, SAT_POS_COLS, SAT_VEL_COLS
import internals.gnss_positioning as gp
import internals.common as common

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
    truth_positions_lla = np.zeros((N, 3))
    truth_positions_ecef = np.zeros((N, 3))
    truth_speed = np.zeros((N,))
    estimated_positions_lla = np.zeros((N, 3))
    estimated_positions_ecef = np.zeros((N, 3))
    estimated_speed = np.zeros((N,))
    estimated_positions_baseline_lla = np.zeros((N, 3))
    estimated_positions_baseline_ecef = np.zeros((N, 3))
    estimated_speed_baseline = np.zeros((N,))
    pseudoranges = []
    sat_positions_ecef = []
    sat_trajectories = {}
    residuals = []

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
        truth_positions_lla[i, :] = pos_truth
        truth_positions_ecef[i, :] = lla_to_ecef(pos_truth.cpu().numpy())
        truth_speed[i] = gt_row['SpeedMps'].to_numpy()
        pseudoranges.append({row['sat_identifier']: row[PR_COL] for _, row in epoch_df.iterrows()})
        sat_positions_ecef.append({row['sat_identifier']: row[SAT_POS_COLS].to_numpy() for _, row in epoch_df.iterrows()})

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

        x = torch.zeros(4, dtype=torch.float32, device='cpu')
        x[0:3] = torch.tensor(truth_positions_ecef[i, :], dtype=torch.float32, device='cpu')
        x[3] = curr_pos["clock_bias"]
        residuals.append(gp.pr_residuals_torch(x, sat_pos, pr - pr_error).cpu().numpy())

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
        estimated_positions_lla[i, :] = est_lla
        estimated_positions_ecef[i, :] = curr_pos["position"].cpu().numpy()
        estimated_speed[i] = np.linalg.norm(curr_pos["velocity"].cpu().numpy())
        estimated_positions_baseline_lla[i, :] = est_lla_baseline
        estimated_positions_baseline_ecef[i, :] = baseline_pos["position"].cpu().numpy()
        estimated_speed_baseline[i] = np.linalg.norm(baseline_pos["velocity"].cpu().numpy())

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

    # RMSE between estimated_positions and truth_positions as well as between estimated_positions_baseline and truth_positions
    # positions are in ecef
    diffs = estimated_positions_ecef - truth_positions_ecef
    diffs_baseline = estimated_positions_baseline_ecef - truth_positions_ecef
    rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    rmse_baseline = np.sqrt(np.mean(np.sum(diffs_baseline**2, axis=1)))
    print(f"RMSE pos 3D (model): {rmse:.2f} m")
    print(f"RMSE pos 3D (baseline): {rmse_baseline:.2f} m")
    print(f"Improvement: {rmse_baseline - rmse:.2f} m, {(rmse_baseline - rmse)/rmse_baseline*100:.2f} %")
    # Horizontal RMSE
    horiz_diffs = diffs[:, :2]
    horiz_diffs_baseline = diffs_baseline[:, :2]
    rmse_horiz = np.sqrt(np.mean(np.sum(horiz_diffs**2, axis=1)))
    rmse_horiz_baseline = np.sqrt(np.mean(np.sum(horiz_diffs_baseline**2, axis=1)))
    print(f"RMSE pos horiz (model): {rmse_horiz:.2f} m")
    print(f"RMSE pos horiz (baseline): {rmse_horiz_baseline:.2f} m")
    print(f"Improvement: {rmse_horiz_baseline - rmse_horiz:.2f} m, {(rmse_horiz_baseline - rmse_horiz)/rmse_horiz_baseline*100:.2f} %")
    # Vertical RMSE
    vert_diffs = diffs[:, 2]
    vert_diffs_baseline = diffs_baseline[:, 2]
    rmse_vert = np.sqrt(np.mean(vert_diffs**2))
    rmse_vert_baseline = np.sqrt(np.mean(vert_diffs_baseline**2))
    print(f"RMSE pos vert (model): {rmse_vert:.2f} m")
    print(f"RMSE pos vert (baseline): {rmse_vert_baseline:.2f} m")
    print(f"Improvement: {rmse_vert_baseline - rmse_vert:.2f} m, {(rmse_vert_baseline - rmse_vert)/rmse_vert_baseline*100:.2f} %")

    # Same for speed
    speed_diffs = estimated_speed - truth_speed
    speed_diffs_baseline = estimated_speed_baseline - truth_speed
    rmse_speed = np.sqrt(np.mean(speed_diffs**2))
    rmse_speed_baseline = np.sqrt(np.mean(speed_diffs_baseline**2))
    print(f"RMSE speed (model): {rmse_speed:.2f} m/s")
    print(f"RMSE speed (baseline): {rmse_speed_baseline:.2f} m/s")
    print(f"Improvement: {rmse_speed_baseline - rmse_speed:.2f} m/s, {(rmse_speed_baseline - rmse_speed)/rmse_speed_baseline*100:.2f} %")
    
    m = folium.Map(
        location=[truth_positions_lla[0,0], truth_positions_lla[0,1]],
        zoom_start=15,
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

    # timestamps = df.groupby("epoch_id")["utcTimeMillis"].mean().apply(lambda ms: pd.to_datetime(ms, unit='ms')).to_numpy()
    # geo_features = []

    # for epoch_idx, timestamp in enumerate(timestamps):
    #     truth_ecef = truth_positions_ecef[epoch_idx]  # shape (3,)

    #     for sv_id, sat_ecef in sat_positions_ecef[epoch_idx].items():
    #         if sv_id not in pseudoranges[epoch_idx]:
    #             continue

    #         pseudorange_m = pseudoranges[epoch_idx][sv_id]
    #         sat_ecef = np.array(sat_ecef, dtype=float).flatten()  # ensure proper shape

    #         los = truth_ecef - sat_ecef
    #         los_norm = np.linalg.norm(los)
    #         if los_norm <= 0.0:
    #             continue

    #         unit_los = los / los_norm
    #         delta_m = pseudorange_m - los_norm
    #         endpoint_ecef = truth_ecef + delta_m * unit_los

    #         sat_lla = ecef_to_lla(sat_ecef)
    #         end_lla = ecef_to_lla(endpoint_ecef)

    #         if np.isnan(sat_lla).any() or np.isnan(end_lla).any():
    #             continue  # skip invalid points

    #         geo_features.append({
    #             "type": "Feature",
    #             "geometry": {
    #                 "type": "LineString",
    #                 "coordinates": [
    #                     [sat_lla[1], sat_lla[0]],
    #                     [end_lla[1], end_lla[0]],
    #                 ],
    #             },
    #             "properties": {
    #                 "time": pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ"),
    #                 "popup": (
    #                     f"SV: {sv_id}<br>"
    #                     f"Geom range: {los_norm:.3f} m<br>"
    #                     f"Pseudorange: {pseudorange_m:.3f} m<br>"
    #                     f"Bias: {pseudorange_m - los_norm:.3f} m"
    #                 ),
    #                 "style": {"color": "orange", "weight": 2, "opacity": 0.7},
    #             },
    #         })

    # TimestampedGeoJson(
    #     {
    #         "type": "FeatureCollection",
    #         "features": geo_features,
    #     },
    #     period="PT1S",
    #     duration="PT1S",
    #     add_last_point=False,
    #     auto_play=False,
    #     loop=False,
    #     max_speed=1,
    #     loop_button=True,
    #     time_slider_drag_update=True,
    # ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(save_name)
    print(f"Saved {save_name}")

    # Residuals scatter plot
    residuals_list = np.array(residuals, dtype=object)
    all_residuals = np.concatenate(residuals_list)

    # X-axis: epoch index repeated for each residual
    epoch_indices = np.concatenate([
        np.full(len(r), i) for i, r in enumerate(residuals_list)
    ])

    plt.figure(figsize=(10, 5))
    plt.scatter(epoch_indices, all_residuals, alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Residual")
    plt.title("Residuals per Epoch")
    plt.grid(True)
    plt.show()
    plt.savefig("pseudorange_residuals.png")