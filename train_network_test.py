import numpy as np
import pandas as pd
from utils.data_file_iter import Data_file_iter
import utils.gnss_positioning as gp
from utils.coord_systems import lla_to_ecef, ecef_to_lla
from network_test import Satellite_weight_network
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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
    base_path = "./smartphone-decimeter-2022"
    data_iter = Data_file_iter(base_path, split='train')

    dataset_stats = pd.read_csv(f"{base_path}/dataset_stats.csv")

    net = Satellite_weight_network().to(device)
    for name, param in net.named_parameters():
        print(name, param.requires_grad)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for df in data_iter:
        print(f"Pre-processing file: {data_iter.get_current_file()}")

        # Add epoch column (group by utcTimeMillis that are within 1s of each other)
        first_time_stamp = df['utcTimeMillis'].iloc[0]
        df['epoch_id'] = ((df['utcTimeMillis'] - first_time_stamp) / 1000).round().astype(int)
        # Put epochs into at most groups of 50 to form a batch.
        # 50 epochs should be enough to capture satellite trends over time.
        # Note: Last batch may have less than 50 epochs.
        df['batch_id'] = df['epoch_id'] // 50

        # Exclude epochs with less than 4 satellites (cannot compute position)
        valid_epochs = df.groupby('epoch_id').filter(lambda x: len(x) >= 4)
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

        # Remove non-L1 rows
        df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]
        df.sort_values(['utcTimeMillis', 'ConstellationType', 'Svid'], inplace=True)

        # Parametric rejection
        df = df[(df['Cn0DbHz'] > 30) & (df['SvElevationDegrees'] > 5)]

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
        df['cumulative_Cn0DbHz_linear'] = (
            df.groupby(['ConstellationType', 'Svid', 'sighting_id'])['Cn0DbHz_linear']
            .cumsum()
        )
        df["mean_Cn0DbHz_linear"] = df['cumulative_Cn0DbHz_linear'] / df['window_size']
        df['var_Cn0DbHz_linear'] = (
            df.groupby(['ConstellationType', 'Svid', 'sighting_id'])['Cn0DbHz_linear']
            .apply(lambda x: x.expanding(min_periods=1).var(ddof=0))
        )

        df['window_size'] /= max_window_size

        df['pseudorange_double_diff'] = double_diff(df, "CorrectedPseudorange")
        df['cn0_double_diff'] = double_diff(df, "Cn0DbHz_linear")

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

        features = [
            'Cn0DbHz_linear',          # linearised signal strength
            'mean_Cn0DbHz_linear',     # mean linearised signal strength
            'var_Cn0DbHz_linear',      # variance of linearised signal strength
            'window_size',             # window size to compute above
            'sin_elevation',           # sat geometry
            'cos_elevation',           # sat geometry
            'sin_azimuth',             # sat geometry
            'cos_azimuth',             # sat geometry
            'seen_t_minus_1',          # binary flag
            'age_since_last_obs',      # scalar
            'residual',                # pseudorange residual (from WLS)
            'pseudorange_double_diff', # time based pseudorange double difference
            'cn0_double_diff',         # time based cn0 double difference
        ]
        batches = []
        lengths = []
        pos_truths = []
        epoch_batch_ids = []

        ground_truth_df = data_iter.get_truth_file()

        # iterate batches (B)
        for batch_id, batch_df in df.groupby('batch_id', sort=True):
            epoch_list = []
            epoch_lengths = []
            epoch_truth_list = []
            epoch_ids = []

            # iterate epochs within batch (variable N)
            for epoch_id, epoch_df in batch_df.groupby('epoch_id', sort=True):
                gt_row = ground_truth_df.iloc[(ground_truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
                pos = gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten()
                epoch_truth_list.append(lla_to_ecef(pos))
                feats = epoch_df[features].to_numpy(dtype=np.float32)
                t = torch.tensor(feats, dtype=torch.float32, device=device)  # (N_i_j, F)
                epoch_list.append(t)
                epoch_lengths.append(t.shape[0])
                epoch_ids.append(epoch_id)

            batches.append(epoch_list)
            lengths.append(epoch_lengths)
            pos_truths.append(epoch_truth_list)
            epoch_batch_ids.append(epoch_ids)
        
        print(f"Done pre-processing, starting training on {len(batches)} batches")
        
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

        for i in range(len(batches)):
            weights_batch = net(batches[i])

            positions_batch = []
            positions_baseline_batch = []
            for j in range(len(weights_batch)):
                pr = torch.tensor(df[df['epoch_id'] == epoch_batch_ids[i][j]]['CorrectedPseudorange'].to_numpy(), dtype=torch.float32)
                prr = torch.tensor(df[df['epoch_id'] == epoch_batch_ids[i][j]]['CorrectedPseudorangeRateMetersPerSecond'].to_numpy(), dtype=torch.float32)
                sat_pos = torch.tensor(df[df['epoch_id'] == epoch_batch_ids[i][j]][['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy(), dtype=torch.float32)
                sat_vel = torch.tensor(df[df['epoch_id'] == epoch_batch_ids[i][j]][['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy(), dtype=torch.float32)

                baseline_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)
                curr_pos = gp.position_torch(pr, prr, sat_pos, sat_vel, Wx=weights_batch[j], prev_estimate=curr_pos)
                positions_batch.append(curr_pos['position'])
                positions_baseline_batch.append(baseline_pos['position'])
            
            positions_batch = torch.stack(positions_batch).to(torch.float32)  # (B, 3)
            positions_baseline_batch = torch.stack(positions_baseline_batch).to(torch.float32)  # (B, 3)
            truth_positions = torch.tensor(pos_truths[i], dtype=torch.float32)

            # Compute loss (e.g., RMSE between predicted and truth positions)
            loss = torch.sqrt(torch.mean((positions_batch - truth_positions) ** 2))
            baseline_loss = torch.sqrt(torch.mean((positions_baseline_batch - truth_positions) ** 2))
            print(f"Batch {i+1} loss: {loss.item():.3f} meters, baseline: {baseline_loss.item():.3f} meters, diff: {baseline_loss.item() - loss.item():.3f}")

            # Backpropagation and optimizer step would go here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            curr_pos = {
                'position': curr_pos['position'].detach(),
                'velocity': curr_pos['velocity'].detach(),
                'clock_bias': curr_pos['clock_bias'].detach(),
                'clock_drift': curr_pos['clock_drift'].detach()
            }
    
    torch.save(net.state_dict(), "weight_network.pt")


