from data_file_iter import Data_file_iterator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from network_test import Satellite_weight_network
from utils import gnss_positioning as gp
from utils.coord_systems import ecef_to_lla, lla_to_ecef
import folium
import inspect

from typing import Any
df: Any = ...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

L1_MIN = 1.55e9
L1_MAX = 1.61e9

def double_diff(df: pd.DataFrame, col: str) -> pd.Series:
    df[f'{col}_epoch_diff'] = df.groupby(['ConstellationType', 'Svid', 'sighting_id'])[col].transform(
            lambda x: x - x.shift()
        ).fillna(0)
    df[f'{col}_epoch_diff_mean'] = (
        df.groupby(['ConstellationType', 'epoch_id'])[f'{col}_epoch_diff']
        # .transform(lambda x: x.abs().mean())
        .transform(lambda x: x.mean())
    )
    df[f'{col}_epoch_diff_var'] = (
        df.groupby(['ConstellationType', 'epoch_id'])[f'{col}_epoch_diff']
        # .transform(lambda x: x.abs().mean())
        .transform(lambda x: x.var())
    )
    df[f'{col}_double_diff'] = np.sqrt((df[f'{col}_epoch_diff'] - df[f'{col}_epoch_diff_mean'])**2 + df[f'{col}_epoch_diff_var'])
    df[f'{col}_double_diff'] = (
        df.groupby(['ConstellationType', 'epoch_id'])[f'{col}_double_diff']
        .transform(lambda x: x / x.abs().max())
    ).fillna(0)
    df.loc[df[f'{col}_epoch_diff'] == 0, f'{col}_double_diff'] = 0
    return df[f'{col}_double_diff']

if __name__ == "__main__":
    # For each data file
    # Split into epochs,
    # Apply data transformations like when training
    # Calculate all the required inputs for the current epoch
    # Predict weights for current epoch
    # Run WLS with those weights
    # Plot position
    net = Satellite_weight_network(input_dim=7)
    net.load_state_dict(torch.load("weight_network.pt"))
    net.eval()
    
    base_path = "./smartphone-decimeter-2023/sdc2023"
    data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
    dataset_stats = pd.read_csv(f"{base_path}/dataset_stats.csv")

    for df in data_iter: 
        #2023-09-06-00-01-us-ca-routen\pixel6pro\device_gnss.csv 
        #2022-02-24-18-29-us-ca-lax-o\mi8\device_gnss.csv 
        #2022-04-01-18-22-us-ca-lax-t\pixel6pro\device_gnss.csv
        #2021-07-14-20-50-us-ca-mtv-e\sm-g988b\device_gnss_preprocessed.csv
        print(f"Pre-processing file: {data_iter.get_current_file()}")

        # Add epoch column (group by utcTimeMillis that are within 1s of each other)
        first_time_stamp = df['utcTimeMillis'].iloc[0]
        df['epoch_id'] = ((df['utcTimeMillis'] - first_time_stamp) / 1000).round().astype(int)
        # Put epochs into at most groups of 50 to form a batch.
        # 50 epochs should be enough to capture satellite trends over time.
        # Note: Last batch may have less than 50 epochs.
        df['batch_id'] = df['epoch_id'] // 50


        # Parametric rejection
        # df = df[(df['Cn0DbHz'] > 30) & (df['SvElevationDegrees'] > 5)]
        
        # Remove non-L1 rows
        df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]

        # Exclude epochs with less than 4 satellites (cannot compute position)
        valid_epochs = df.groupby('epoch_id').filter(lambda x: len(x) >= 5) # 5 to be able to compute n-1 residuals
        df = valid_epochs

        # Drop rows with missing essential data
        df = df.dropna(subset=[
            'RawPseudorangeMeters',
            'PseudorangeRateMetersPerSecond',
            'SvPositionXEcefMeters',
            'SvPositionYEcefMeters',
            'SvPositionZEcefMeters',
            'SvVelocityXEcefMetersPerSecond',
            'SvVelocityYEcefMetersPerSecond',
            'SvVelocityZEcefMetersPerSecond',
            'SvElevationDegrees',
            'SvAzimuthDegrees',
            'Cn0DbHz'
        ])
        df.sort_values(['utcTimeMillis', 'ConstellationType', 'Svid'], inplace=True)

        # Apply iono, tropo and sv clock corrections to pseudorange
        df['CorrectedPseudorange'] = df['RawPseudorangeMeters'] - df['IonosphericDelayMeters'] - df['TroposphericDelayMeters'] + df['SvClockBiasMeters']
        df['CorrectedPseudorangeRateMetersPerSecond'] = df['PseudorangeRateMetersPerSecond'] + df['SvClockDriftMetersPerSecond']

        # Linearise and normalise Cn0DbHz column
        df['Cn0DbHz_linear'] = 10 ** (df['Cn0DbHz'] / 10)
        df['Cn0DbHz_linear'] /= dataset_stats.loc[dataset_stats['column_name'] == 'Cn0DbHz_linear', 'abs_max_value'].values[0]

        df["sin_elevation"] = np.sin(np.deg2rad(df['SvElevationDegrees']))
        df["cos_elevation"] = np.cos(np.deg2rad(df['SvElevationDegrees']))
        df["sin_azimuth"] = np.sin(np.deg2rad(df['SvAzimuthDegrees']))
        df["cos_azimuth"] = np.cos(np.deg2rad(df['SvAzimuthDegrees']))

        # Add seen_t_minus_1 and age_since_last_obs and lock_time columns
        # Since sivd is not unique, it needs grouping with constellation type to be able to identify unique satellites
        df = df.sort_values(['epoch_id', 'ConstellationType', 'Svid'])
        df['prev_epoch_seen'] = df.groupby(['ConstellationType', 'Svid'])['epoch_id'].shift()
        df["seen_t_minus_1"] = df["epoch_id"] - df["prev_epoch_seen"] == 1
        df['sighting_id'] = df.groupby(['ConstellationType', 'Svid'])['seen_t_minus_1'].transform(lambda x: (~x).cumsum() - 1)
        df['age_since_last_obs'] = (df['epoch_id'] - df['prev_epoch_seen']).fillna(-1)
        df['lock_time'] = (
            df.groupby(['ConstellationType', 'Svid', 'sighting_id'])['seen_t_minus_1']
            .transform(lambda x: x.cumsum())
        )
        max_window_size = 10
        df["window_size"] = (df["lock_time"] + 1).clip(upper=max_window_size)
        # df['cumulative_Cn0DbHz_linear'] = (
        #     df.groupby(['ConstellationType', 'Svid', 'sighting_id'])['Cn0DbHz_linear']
        #     .cumsum()
        # )
        # df["mean_Cn0DbHz_linear"] = df['cumulative_Cn0DbHz_linear'] / df['window_size']
        # df['var_Cn0DbHz_linear'] = (
        #     df.groupby(['ConstellationType', 'Svid', 'sighting_id'])['Cn0DbHz_linear']
        #     .apply(lambda x: x.expanding(min_periods=1).var(ddof=0))
        # )

        def compute_satellite_windows(group: pd.DataFrame) -> pd.DataFrame:
            mean_cn0_linear_list = []
            var_cn0_linear_list = []
            var_cn0_list = []
            for i in range(len(group)):
                w = group.iloc[i]['window_size']
                start_idx = max(0, i - w + 1)
                window_linear = group.iloc[start_idx:i+1]['Cn0DbHz_linear']
                window = group.iloc[start_idx:i+1]['Cn0DbHz']
                mean_cn0_linear_list.append(window_linear.mean())
                var_cn0_linear_list.append(window_linear.var(ddof=0))  # population variance
                var_cn0_list.append(window.var(ddof=0))

            group['mean_Cn0DbHz_linear'] = mean_cn0_linear_list
            group['var_Cn0DbHz_linear'] = var_cn0_linear_list
            group['var_Cn0DbHz'] = var_cn0_list
            return group
        
        df = df.groupby(['ConstellationType', 'Svid'], group_keys=False).apply(compute_satellite_windows, include_groups=True)

        df['window_size'] /= max_window_size

        df['pseudorange_double_diff'] = double_diff(df, "CorrectedPseudorange")
        df['cn0_double_diff'] = double_diff(df, "Cn0DbHz_linear")

        df['residual_matrix'] = None
        df['residual_matrix'] = df['residual_matrix'].astype(object)

        curr_pos = None
        for epoch_id, epoch_df in df.groupby('epoch_id'):
            # Get pseudoranges and satellite positions
            pr = epoch_df['CorrectedPseudorange'].to_numpy()
            prr = epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
            sat_pos = epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
            sat_vel = epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

            # Compute receiver position using least squares with no weighting
            curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)

            # Compute residual for all included satellites
            x = np.zeros(4)
            x[:3] = curr_pos["position"]
            x[3] = curr_pos["clock_bias"]
            res = gp.pr_residuals(x, sat_pos, pr)
            res = res / abs(res).mean()

            df.loc[epoch_df.index, 'residual'] = res
            df.loc[epoch_df.index, 'pr_unc_norm'] = epoch_df['RawPseudorangeUncertaintyMeters'] / np.max(np.abs(epoch_df['RawPseudorangeUncertaintyMeters']))

            N = len(epoch_df)
            res_matrix = np.full((N, N-1), np.nan)
            # For each satellite, exclude it and compute position with remaining satellites
            i = 0
            for idx, sat_row in epoch_df.iterrows():
                excluded_sat = (sat_row['ConstellationType'], sat_row['Svid'])
                included_sats = epoch_df[
                    ~((epoch_df['ConstellationType'] == excluded_sat[0]) & (epoch_df['Svid'] == excluded_sat[1]))
                ]
                assert included_sats.shape[0] == epoch_df.shape[0] - 1, "Only one satellite should be excluded"

                # Get pseudoranges and satellite positions
                pr = included_sats['CorrectedPseudorange'].to_numpy()
                prr = included_sats['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
                sat_pos = included_sats[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
                sat_vel = included_sats[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

                # Compute receiver position using least squares with no weighting
                curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)

                # Compute residual for all included satellites
                x = np.zeros(4)
                x[:3] = curr_pos["position"]
                x[3] = curr_pos["clock_bias"]
                res = gp.pr_residuals(x, sat_pos, pr)
                df.at[idx, 'residual_matrix'] = res / abs(res).mean()

        features = [
            'Cn0DbHz_linear',          # linearised signal strength
            # 'mean_Cn0DbHz_linear',     # mean linearised signal strength
            # 'var_Cn0DbHz_linear',      # variance of linearised signal strength
            # 'window_size',             # window size to compute above
            'sin_elevation',           # sat geometry
            'cos_elevation',           # sat geometry
            # 'sin_azimuth',             # sat geometry
            # 'cos_azimuth',             # sat geometry
            'seen_t_minus_1',          # binary flag
            'age_since_last_obs',      # scalar
            'residual',                # pseudorange residual (from WLS)
            # 'pseudorange_double_diff', # time based pseudorange double difference
            # 'cn0_double_diff',         # time based cn0 double difference
            'pr_unc_norm'              # normalized pseudorange uncertainty
        ]

        ground_truth_df = data_iter.get_truth_file()

        sat_trajectories = {}

        curr_pos = None
        # num unique epochs
        N = df['epoch_id'].nunique()
        truth_positions = np.zeros((N, 3))
        estimated_positions = np.zeros((N, 3))
        estimated_positions_baseline = np.zeros((N, 3))
        for epoch_id, epoch_df in df.groupby('epoch_id', sort=True):
            gt_row = ground_truth_df.iloc[(ground_truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
            pos_truth = gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten()
            truth_positions[epoch_id, :] = pos_truth
            feats = epoch_df[features].to_numpy(dtype=np.float32)
            t = [torch.tensor(feats, dtype=torch.float32, device=device)]  # (N_i_j, F)
            res_mats = [torch.tensor(np.stack(epoch_df['residual_matrix'].to_numpy(), dtype=np.float32), dtype=torch.float32, device=device)]
            with torch.no_grad():
                weights_diag, pr_corrections = net(t, res_mats)  # list of (N_i_j, N_i_j) diagonal matrices

            pr = torch.tensor(epoch_df['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
            pr = pr + pr_corrections[0]
            prr = torch.tensor(epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
            sat_pos = torch.tensor(epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
            sat_vel = torch.tensor(epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)
            curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=weights_diag[0], prev_estimate=curr_pos)
            # baseline weights are 1/var(cn0) per satellite
            pr_unc = epoch_df['RawPseudorangeUncertaintyMeters'].to_numpy()

            # Avoid division by zero or massive weights
            baseline_weights = 1.0 / (pr_unc + 1e-6)

            # Rescale to mean 1 for stability
            baseline_weights /= np.mean(baseline_weights)

            # Add a floor (so no weight is too small)
            baseline_weights = np.maximum(baseline_weights, 0.05)

            baseline_weights_diag = torch.diag(torch.tensor(baseline_weights, dtype=torch.float32))
            pos_baseline = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=baseline_weights_diag, prev_estimate=curr_pos)
            est_lla = ecef_to_lla(curr_pos["position"].cpu().numpy())
            est_lla_baseline = ecef_to_lla(pos_baseline["position"].cpu().numpy())
            estimated_positions[epoch_id, :] = est_lla
            estimated_positions_baseline[epoch_id, :] = est_lla_baseline

            for _, row in epoch_df.iterrows():
                sv_id = row["ConstellationType"] + '_' + row["Svid"]  # or row["ConstellationType"] + row["Svid"] if mixing GNSS constellations
                key = f"{sv_id}"

                if key not in sat_trajectories:
                    sat_trajectories[key] = []

                sv_ecef = np.array([
                    row["SvPositionXEcefMeters"],
                    row["SvPositionYEcefMeters"],
                    row["SvPositionZEcefMeters"],
                ])

                sv_lat, sv_lon, sv_alt = ecef_to_lla(sv_ecef)
                sat_trajectories[key].append((sv_lat, sv_lon))

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

        m.save("trajectory_map.html")
        break