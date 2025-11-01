import numpy as np
import pandas as pd
from utils.data_file_iter import Data_file_iter
import utils.gnss_positioning as gp
from utils.coord_systems import lla_to_ecef, ecef_to_lla
from network import Network
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

L1_MIN = 1.55e9
L1_MAX = 1.61e9

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2022"
    data_iter = Data_file_iter(base_path, split='train')

    dataset_stats = pd.read_csv(f"{base_path}/dataset_stats.csv")

    net = Network(78).to(device)
    for name, param in net.named_parameters():
        print(name, param.requires_grad)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for df in data_iter:
        print(f"Pre-processing file: {data_iter.get_current_file()}")
        # Remove non-L1 rows (CodeType == C for L1)
        df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]
        df.sort_values(['utcTimeMillis', 'ConstellationType', 'Svid'], inplace=True)

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

        # Apply iono, tropo and sv clock corrections to pseudorange
        df['CorrectedPseudorange'] = df['RawPseudorangeMeters'] - df['IonosphericDelayMeters'] - df['TroposphericDelayMeters'] + df['SvClockBiasMeters']
        df['CorrectedPseudorangeRateMetersPerSecond'] = df['PseudorangeRateMetersPerSecond'] + df['SvClockDriftMetersPerSecond']

        # Linearise and normalise Cn0DbHz column
        df['Cn0DbHz_linear'] = 10 ** (df['Cn0DbHz'] / 10)
        df['Cn0DbHz_linear'] /= dataset_stats.loc[dataset_stats['column_name'] == 'Cn0DbHz_linear', 'abs_max_value'].values[0]

        # Add epoch column (group by utcTimeMillis that are within 1s of each other)
        first_time_stamp = df['utcTimeMillis'].iloc[0]
        df['epoch_id'] = ((df['utcTimeMillis'] - first_time_stamp) / 1000).round().astype(int)


        # epoch_ids = (df['utcTimeMillis'].diff().fillna(0).astype(np.int64) >= 1000).cumsum()
        # df['epoch_id'] = epoch_ids

        # Add seen_t_minus_1 and age_since_last_obs and lock_time columns
        # Since sivd is not unique, it needs grouping with constellation type to be able to identify unique satellites
        df = df.sort_values(['epoch_id', 'ConstellationType', 'Svid'])
        df['prev_epoch_seen'] = df.groupby(['ConstellationType', 'Svid'])['epoch_id'].shift()
        df["seen_t_minus_1"] = df["epoch_id"] - df["prev_epoch_seen"] == 1
        df['age_since_last_obs'] = (df['epoch_id'] - df['prev_epoch_seen']).fillna(-1)
        df['lock_time'] = df.groupby(['ConstellationType', 'Svid'])['seen_t_minus_1'].transform(
            lambda x: np.cumsum(x & (x.shift(fill_value=False)))
        )

        # Put epochs into at most groups of 50 to form a batch.
        # 50 epochs should be enough to capture satellite trends over time.
        # Note: Last batch may have less than 50 epochs.
        df['batch_id'] = df['epoch_id'] // 50

        # Exclude epochs with less than 4 satellites (cannot compute position)
        valid_epochs = df.groupby('epoch_id').filter(lambda x: len(x) >= 5) # However to compute N-1 residuals we need at least 5 sats
        df = valid_epochs

        batches = []
        last_pr = {}  # key: (ConstellationType, Svid) -> float. Used for double difference
        curr_pos = None
        # For each epoch in a batch, construct the residual matrix and the other features
        for batch_id, batch_df in df.groupby('batch_id'):
            batch_df = batch_df.sort_values('epoch_id')
            epoch_data = []
            for epoch_id, epoch_df in batch_df.groupby('epoch_id'):
                # Calculate the pseudorange epoch difference for satellites that were seen in the previous epoch
                # Note: double difference is only defined for satellite in the same constellation
                for constellation in epoch_df['ConstellationType'].unique():
                    constel_epoch_df = epoch_df[epoch_df['ConstellationType'] == constellation]

                    # Only satellites that were seen in the previous epoch
                    sats_seen_prev_epoch = constel_epoch_df[constel_epoch_df['seen_t_minus_1']]
                    # Set pseudorange_double_diff to 0 for satellites not seen in previous epoch
                    sats_not_seen_prev_epoch = constel_epoch_df[~constel_epoch_df['seen_t_minus_1']]
                    epoch_df.loc[sats_not_seen_prev_epoch.index, 'pseudorange_double_diff'] = 0.0

                    if sats_seen_prev_epoch.empty:
                        continue

                    delta_pr = []
                    valid_indices = []

                    # Compute x(t) - x(t-1) for each satellite
                    for idx, row in sats_seen_prev_epoch.iterrows():
                        key = (row['ConstellationType'], row['Svid'])
                        if key in last_pr:
                            delta = row['CorrectedPseudorange'] - last_pr[key]
                            delta_pr.append(delta)
                            valid_indices.append(idx)

                    delta_pr = np.array(delta_pr)
                    if len(delta_pr) == 0:
                        continue

                    # Compute mean and variance across satellites
                    mu = np.mean(delta_pr)
                    sigma2 = np.var(delta_pr)

                    # Compute PDD scores for each satellite
                    pdd_scores = np.sqrt((delta_pr - mu)**2 + sigma2)

                    # Store in the DataFrame
                    epoch_df.loc[valid_indices, 'pseudorange_double_diff'] = pdd_scores / 1e3

                # Update last_pr for all satellites in this epoch
                for idx, row in epoch_df.iterrows():
                    key = (row['ConstellationType'], row['Svid'])
                    last_pr[key] = row['CorrectedPseudorange']

                # # Optional: assign updated epoch_df back to main df
                # df.loc[epoch_df.index, 'pseudorange_double_diff'] = epoch_df['pseudorange_double_diff']

                residuals = []
                # For each satellite, exclude it and compute position with remaining satellites
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
                    residuals.append(res / abs(res).mean())
                assert max(len(r) for r in residuals) == min(len(r) for r in residuals), "All residual arrays must be the same length"

                # Construct tensor of other features
                epoch_df["sin_elevation"] = np.sin(np.deg2rad(epoch_df['SvElevationDegrees']))
                epoch_df["cos_elevation"] = np.cos(np.deg2rad(epoch_df['SvElevationDegrees']))
                epoch_df["sin_azimuth"] = np.sin(np.deg2rad(epoch_df['SvAzimuthDegrees']))
                epoch_df["cos_azimuth"] = np.cos(np.deg2rad(epoch_df['SvAzimuthDegrees']))

                pr = epoch_df['CorrectedPseudorange'].to_numpy()
                prr = epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
                sat_pos = epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
                sat_vel = epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()
                curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)

                epoch_df['dop_pdop'] = curr_pos['dop']['pdop']
                epoch_df['dop_hdop'] = curr_pos['dop']['hdop']
                epoch_df['dop_vdop'] = curr_pos['dop']['vdop']
                epoch_df['dop_tdop'] = curr_pos['dop']['tdop']
                epoch_df['dop_gdop'] = curr_pos['dop']['gdop']

                # One-hot encode constellation type (max 7 types)
                constellation_one_hot = pd.get_dummies(epoch_df['ConstellationType'], prefix='constel')
                constellation_one_hot = constellation_one_hot.reindex(columns=[f'constel_{i}' for i in range(7)], fill_value=0)
                epoch_df = pd.concat([epoch_df, constellation_one_hot], axis=1)

                epoch_df["seen_t_minus_1"] = epoch_df["seen_t_minus_1"].astype(float)

                other_features = epoch_df[[
                    'seen_t_minus_1',
                    'age_since_last_obs',
                    'dop_pdop',
                    'dop_hdop',
                    'dop_vdop',
                    'dop_tdop',
                    'dop_gdop',
                    'sin_elevation',
                    'cos_elevation',
                    'sin_azimuth',
                    'cos_azimuth',
                    'lock_time',
                    # 'Cn0DbHz_linear',  # mean
                    # 'Cn0DbHz_linear',  # var
                    'Cn0DbHz_linear',  # current
                    'pseudorange_double_diff'
                ]].to_numpy().astype(np.float32)
                epoch_tensor = torch.tensor(other_features, dtype=torch.float32)
                timestamp = epoch_df['utcTimeMillis'].iloc[0]
                epoch_data.append((timestamp, epoch_id, residuals, epoch_tensor))
            
            batches.append(epoch_data)

        print(f"Done pre-processing, starting training on {len(batches)} batches")

        ground_truth_df = data_iter.get_truth_file()

        curr_pos = None

        # Initial position estimate using first epoch otherwise gradients are terrible initially
        pr = df[df['epoch_id'] == 0]['CorrectedPseudorange'].to_numpy()
        prr = df[df['epoch_id'] == 0]['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
        sat_pos = df[df['epoch_id'] == 0][['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
        sat_vel = df[df['epoch_id'] == 0][['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

        curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)
        curr_pos["position"] = torch.tensor(curr_pos["position"], dtype=torch.float32)
        curr_pos["clock_bias"] = torch.tensor(curr_pos["clock_bias"], dtype=torch.float32)
        curr_pos["velocity"] = torch.tensor(curr_pos["velocity"], dtype=torch.float32)
        curr_pos["clock_drift"] = torch.tensor(curr_pos["clock_drift"], dtype=torch.float32)

        for (batch_id, batch) in enumerate(batches):
            timestamps = [item[0] for item in batch]
            epoch_ids = [item[1] for item in batch]
            residuals_batch = [item[2] for item in batch]
            other_data_batch = [item[3] for item in batch]

            # Work out what will be padded so it can be unpadded later
            sat_counts = [data.shape[0] for data in other_data_batch]

            # # Pad other data batch to create a batch tensor
            # padded_other_data = nn.utils.rnn.pad_sequence(other_data_batch, batch_first=True)  # (B, N_max, F)
            # # for i in range(len(residuals_batch)):
            # for i, res_list in enumerate(residuals_batch):
            #     # residuals_batch[i] = nn.utils.rnn.pad_sequence(torch.Tensor(residuals_batch[i]), batch_first=True)  # (N, N)
            #     # Find max length needed for padding
            #     N = len(res_list)  # Number of satellites
            #     # Pad each residual array to length N 
            #     padded_res = []
            #     for res in res_list:
            #         # Create zero-padded array of size N
            #         padded = np.zeros(N, dtype=np.float32)
            #         padded[:len(res)] = res  # Copy original values
            #         padded_res.append(padded)
            #     arr = np.stack(padded_res, axis=0)  # (N, N)
            #     residuals_batch[i] = torch.from_numpy(arr).float()

            # Truth positions for the batch
            # Note: timestamps do not directly line up, so the closest timestamp in ground truth is used
            truth_positions = []
            for ts in timestamps:
                gt_row = ground_truth_df.iloc[(ground_truth_df['UnixTimeMillis'] - ts).abs().argsort()[:1]]
                pos = gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten()
                truth_positions.append(lla_to_ecef(pos))

            truth_positions = torch.tensor(truth_positions, dtype=torch.float32)  # (B, 3)

            # Forward pass through the network
            # weights = net(residuals_batch, padded_other_data)
            weights = net(residuals_batch, other_data_batch)

            # Unpad weights to original satellite counts
            # unpadded_weights = []
            # for i, count in enumerate(sat_counts):
            #     unpadded_weights.append(weights[i, :count, :count])
            # weights = unpadded_weights  # List of (N, N) weight matrices
            # weights = torch.stack(weights)  # (B, N, N)

            positions_batch = []
            positions_baseline_batch = []
            for i in range(len(residuals_batch)):
                pr = torch.tensor(df[df['epoch_id'] == epoch_ids[i]]['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
                prr = torch.tensor(df[df['epoch_id'] == epoch_ids[i]]['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
                sat_pos = torch.tensor(df[df['epoch_id'] == epoch_ids[i]][['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
                sat_vel = torch.tensor(df[df['epoch_id'] == epoch_ids[i]][['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)

                count = sat_pos.shape[0]
                baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)
                curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=weights[i], prev_estimate=curr_pos)
                positions_batch.append(curr_pos['position'])
                positions_baseline_batch.append(baseline_pos['position'])

            positions_batch = torch.stack(positions_batch).to(torch.float32)  # (B, 3)
            positions_baseline_batch = torch.stack(positions_baseline_batch).to(torch.float32)  # (B, 3)

            # Compute loss (e.g., RMSE between predicted and truth positions)
            loss = torch.sqrt(torch.mean((positions_batch - truth_positions) ** 2))
            baseline_loss = torch.sqrt(torch.mean((positions_baseline_batch - truth_positions) ** 2))
            print(f"Batch {batch_id+1} loss: {loss.item():.3f} meters, baseline: {baseline_loss.item():.3f} meters, diff: {baseline_loss.item() - loss.item():.3f}")

            # Backpropagation and optimizer step would go here
            optimizer.zero_grad()
            loss.backward()

            curr_pos = {
                'position': curr_pos['position'].detach(),
                'velocity': curr_pos['velocity'].detach(),
                'clock_bias': curr_pos['clock_bias'].detach(),
                'clock_drift': curr_pos['clock_drift'].detach()
            }

    torch.save(net, "weight_network.nwk")
        



        

