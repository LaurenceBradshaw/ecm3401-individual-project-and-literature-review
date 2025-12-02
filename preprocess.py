import pandas as pd
import os
import numpy as np
from data_file_iter import Data_file_iterator
from utils import gnss_positioning as gp
from collections import defaultdict
from scipy.optimize import curve_fit
from utils.coord_systems import lla_to_ecef

# import matplotlib.pyplot as plt

L1_MIN = 1.55e9
L1_MAX = 1.61e9
SPEED_OF_LIGHT = 299792458.0  # m/s


if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"
    output_file = f"{base_path}/dataset_stats.csv"
    data_iter = Data_file_iterator(base_path, split='train', preprocessed=False, prefix='2021-07-14-20-50-us-ca-mtv-e')

    for df in data_iter:
        print(f"Pre-processing file: {data_iter.get_current_file()}")
        # first_time_stamp = df['utcTimeMillis'].iloc[0]
        # df['epoch_id'] = ((df['utcTimeMillis'] - first_time_stamp) / 1000).round().astype(int)
        df["epoch_id"] = df.groupby('utcTimeMillis').ngroup()
        # Filter to L1 frequency band only. L5 is a pain
        df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]
        # Drop rows with missing critical data
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
        if df.empty:
            continue
        # Filter out epochs with insufficient satellites
        df = df.groupby('epoch_id').filter(lambda x: len(x) >= 5)

        # Apply iono, tropo and sv clock corrections to pseudorange
        df['CorrectedPseudorange'] = df['RawPseudorangeMeters'] - df['IonosphericDelayMeters'] - df['TroposphericDelayMeters'] - df['IsrbMeters'] + df['SvClockBiasMeters']
        df['CorrectedPseudorangeRateMetersPerSecond'] = df['PseudorangeRateMetersPerSecond'] + df['SvClockDriftMetersPerSecond']

        df['Cn0DbHz_linear'] = 10 ** (df['Cn0DbHz'] / 10)

        df["sin_elevation"] = np.sin(np.deg2rad(df['SvElevationDegrees']))
        df["cos_elevation"] = np.cos(np.deg2rad(df['SvElevationDegrees']))
        df["sin_azimuth"] = np.sin(np.deg2rad(df['SvAzimuthDegrees']))
        df["cos_azimuth"] = np.cos(np.deg2rad(df['SvAzimuthDegrees']))

        df['prev_epoch_seen'] = df.groupby(['ConstellationType', 'Svid'])['epoch_id'].shift()
        df["seen_t_minus_1"] = df["epoch_id"] - df["prev_epoch_seen"] == 1
        df.drop(columns=['prev_epoch_seen'], inplace=True)
        
        sat_pos_cols = ['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']
        sat_vel_cols = ['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']
        max_reasonable_jump_km = 200.0
        max_reasonable_speed_mps = 18000.0
        for (constellation, svid), group in df.groupby(['ConstellationType', 'Svid']):
            g = group.sort_values('utcTimeMillis').copy()
            idx = g.index.to_numpy()

            r = g[sat_pos_cols].to_numpy()
            v = g[sat_vel_cols].to_numpy()
            t = g['utcTimeMillis'].to_numpy()

            # compute position deltas
            dr = np.diff(r, axis=0)
            dt = np.diff(t).astype(float)
            dt[dt <= 0] = np.nan

            jump_dist = np.linalg.norm(dr, axis=1) / 1000.0
            v_est = dr / dt[:, None]
            v_est_mag = np.linalg.norm(v_est, axis=1)

            # reported velocity magnitude
            v_reported_mag = np.linalg.norm(v, axis=1)

            # flag bad epochs (1: end of delta, 0: nothing)
            bad = np.zeros(len(g), dtype=bool)
            bad[1:] |= (jump_dist > max_reasonable_jump_km)
            bad[1:] |= (v_est_mag > max_reasonable_speed_mps)
            bad |= (v_reported_mag > max_reasonable_speed_mps)

            # record initial flags
            df.loc[idx[bad], 'motion_flag'] = 'bad_motion'

            # if any bad points exist, interpolate them
            if bad.any():
                print(f"Found {np.sum(bad)} bad motion instances for satellite {constellation} {svid} in file" \
                      f"{data_iter.get_current_file()}, repairing via interpolation...")
                good_mask = ~bad
                good_times = t[good_mask]

                # positions
                for i, col in enumerate(sat_pos_cols):
                    good_vals = r[good_mask, i]
                    interp_vals = np.interp(t, good_times, good_vals)
                    df.loc[idx, col] = interp_vals

                # velocities (optional but recommended)
                for i, col in enumerate(sat_vel_cols):
                    good_vals = v[good_mask, i]
                    interp_vals = np.interp(t, good_times, good_vals)
                    df.loc[idx, col] = interp_vals
        
        df['residual_matrix'] = None
        df['residual_matrix'] = df['residual_matrix'].astype(object)
        truth_df = data_iter.get_truth_file()
        curr_pos = None
        curr_pos_weighted = None
        sat_samples = {}
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
            df.loc[epoch_df.index, 'residual'] = res

            v = np.zeros(4)
            v[:3] = curr_pos['velocity']
            v[3] = curr_pos['clock_drift']
            rate_res = gp.prr_residuals(v, sat_vel, prr, x, sat_pos)
            df.loc[epoch_df.index, 'rate_residual'] = rate_res

            los = gp.los_vector(curr_pos['position'], sat_pos)[0]
            df.loc[epoch_df.index, 'los_x'] = los[:,0]
            df.loc[epoch_df.index, 'los_y'] = los[:,1]
            df.loc[epoch_df.index, 'los_z'] = los[:,2]

            wavelength = SPEED_OF_LIGHT / epoch_df['CarrierFrequencyHz'].to_numpy()
            rho_geom = np.sum(los * (sat_vel - curr_pos['velocity']), axis=1)
            rho_clk = curr_pos['clock_drift'] - epoch_df["SvClockDriftMetersPerSecond"].to_numpy()
            rho_pred = rho_geom + rho_clk
            D_pred = -(rho_pred / wavelength)
            D_meas = -prr / wavelength
            D_res = D_meas - D_pred

            df.loc[epoch_df.index, 'doppler_residual'] = D_res
            epoch_df.loc[epoch_df.index, 'doppler_residual'] = D_res

            # Baseline weights
            pr_weights_baseline = 1 / ((1/epoch_df['sin_elevation'])**2 + (4*10**(-epoch_df['Cn0DbHz']/30))**2)
            df.loc[epoch_df.index, 'pr_baseline_weight'] = pr_weights_baseline

            prr_weights_baseline = np.zeros(len(epoch_df))
            i=0
    
            for _, row in epoch_df.iterrows():
                sat_key = (row['ConstellationType'], row['Svid'])
                res = row['doppler_residual']
                # a = np.deg2rad(row['SvElevationDegrees'])  # or elevation angle if available
                b = row['Cn0DbHz']  # C/N0
                
                # Initialise per-satellite list if needed
                if sat_key not in sat_samples:
                    sat_samples[sat_key] = []
                
                # Add current residual to satellite history
                sat_samples[sat_key].append(res)
                
                # Use last N samples for batch variance (or all if fewer than N)
                samples = sat_samples[sat_key][-50:]  # e.g., last 50 residuals
                N = len(samples)
                
                if N > 1:
                    s2 = np.var(samples, ddof=1)  # unbiased sample variance
                else:
                    s2 = 1e-3  # default small value for new satellite
                
                # Compute mean geometry and mean noise scale for the batch
                sin2a = 1/row['sin_elevation']**2
                c = 10**(-b/10)
                
                # Estimate k using batch residual formula
                k_hat = max(s2 - sin2a * c, 1e-6)  # clamp to positive
                
                # Compute sigma^2 for weight
                sigma2 = sin2a + k_hat * c
                prr_weights_baseline[i] = 1.0 / sigma2  # weight = 1 / sigma^2
                i+=1
                
            # Save weights back to DataFrame
            df.loc[epoch_df.index, 'prr_baseline_weight'] = prr_weights_baseline

            pr_weights_baseline = np.diag(pr_weights_baseline)
            prr_weights_baseline = np.diag(prr_weights_baseline)
            curr_pos_weighted = gp.position(pr, prr, sat_pos, sat_vel, Wx=pr_weights_baseline, Wv=prr_weights_baseline, prev_estimate=curr_pos_weighted)
            gt_row = truth_df.iloc[(truth_df['UnixTimeMillis'] - epoch_df["utcTimeMillis"].mean()).abs().argsort()[:1]]
            truth_pos = lla_to_ecef(gt_row[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy().flatten())
            error_m = np.linalg.norm(curr_pos_weighted['position'] - truth_pos)
            df.loc[epoch_df.index, 'baseline_position_error_m'] = error_m

            # prr_unc = epoch_df['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy()
            # prr_weights_baseline = 1.0 / (prr_unc + 1e-6)
            # prr_weights_baseline /= np.mean(prr_weights_baseline)
            # prr_weights_baseline = np.maximum(prr_weights_baseline, 0.05)
            # df.loc[epoch_df.index, 'prr_baseline_weight'] = prr_weights_baseline

            # For each satellite, exclude it and compute position with remaining satellites
            # for idx, sat_row in epoch_df.iterrows():
            #     excluded_sat = (sat_row['ConstellationType'], sat_row['Svid'])
            #     included_sats = epoch_df[
            #         ~((epoch_df['ConstellationType'] == excluded_sat[0]) & (epoch_df['Svid'] == excluded_sat[1]))
            #     ]
            #     assert included_sats.shape[0] == epoch_df.shape[0] - 1, f"Only one satellite should be excluded, but got {included_sats.shape[0]} included vs {epoch_df.shape[0]} total."

            #     # Get pseudoranges and satellite positions
            #     pr = included_sats['CorrectedPseudorange'].to_numpy()
            #     prr = included_sats['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
            #     sat_pos = included_sats[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
            #     sat_vel = included_sats[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

            #     # Compute receiver position using least squares with no weighting
            #     curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)

            #     # Compute residual for all included satellites
            #     x = np.zeros(4)
            #     x[:3] = curr_pos["position"]
            #     x[3] = curr_pos["clock_bias"]
            #     res = gp.pr_residuals(x, sat_pos, pr)
            #     df.at[idx, 'residual_matrix'] = res
        
        poor_mask = df['baseline_position_error_m'] > 10.0
        poor_region = poor_mask.to_numpy()
        poor_indices = np.where(poor_mask)[0]
        for idx in poor_indices:
            start = max(0, idx - 10)
            poor_region[start:idx+1] = True
        
        df.loc[df.index, 'poor_region'] = poor_region
        df.to_csv(f"{data_iter.get_current_file().replace(".csv","")}_preprocessed.csv", index=False)

    # --------------------------------------------------------------
    # 3. Fit elevation → C/N0 curves
    # --------------------------------------------------------------

    def model_func(elev, c0, c1):
        return c0 + c1 * np.log(abs(elev) + 1.0)

    samples = defaultdict(list)

    data_iter = Data_file_iterator(base_path, split='train')
    for df in data_iter:
        if {'ConstellationType', 'Svid', 'SvElevationDegrees', 'Cn0DbHz'}.issubset(df.columns):
            elev_vals = df['SvElevationDegrees'].values
            cn0_vals = df['Cn0DbHz'].values
            const_vals = df['ConstellationType'].values
            svid_vals = df['Svid'].values

            mask = (~np.isnan(elev_vals)) & (~np.isnan(cn0_vals)) & (~pd.isna(svid_vals))
            elev_vals = elev_vals[mask]
            cn0_vals = cn0_vals[mask]
            const_vals = const_vals[mask]
            svid_vals = svid_vals[mask]

            for elev, cn0, const, svid in zip(elev_vals, cn0_vals, const_vals, svid_vals):
                key = f"{const}_{svid}"  # unique satellite key
                samples[key].append((float(elev), float(cn0)))

    coefs = {}
    for sat_key, pair_list in samples.items():
        arr = np.array(pair_list)
        elevs = arr[:, 0]
        cn0s = arr[:, 1]

        if len(elevs) < 2:
            # Not enough points to fit two parameters, fallback to constant model
            coefs[sat_key] = (np.mean(cn0s), 0.0)
            print(f"{sat_key}: only {len(elevs)} point(s), using constant model")
            continue

        init_guess = (np.mean(cn0s), 1.0)  # initial guess for c0, c1
        popt, _ = curve_fit(model_func, elevs, cn0s, p0=init_guess, maxfev=20000)
        coefs[sat_key] = (popt[0], popt[1])
        print(f"{sat_key}: c0={popt[0]:.4f}, c1={popt[1]:.4f}")

    data_iter = Data_file_iterator(base_path, split='train')
    for df in data_iter:
        # construct satellite key for each row
        df['sat_key'] = df['ConstellationType'].astype(str) + '_' + df['Svid'].astype(str)

        df['coef_c0'] = df['sat_key'].map(lambda k: coefs[k][0])
        df['coef_c1'] = df['sat_key'].map(lambda k: coefs[k][1])

        expected = model_func(
            df['SvElevationDegrees'].values,
            df['coef_c0'].values,
            df['coef_c1'].values
        )
        df['Cn0DbHz_diff'] = expected - df['Cn0DbHz'].values

        df.to_csv(data_iter.get_current_file(), index=False)



    stats = {}  # column -> {count, mean, M2}
    samples = defaultdict(list)

    def update_welford(col_name, col_values):
        x = col_values.astype(float)
        mask = ~np.isnan(x)
        x = x[mask]

        if len(x) == 0:
            return

        if col_name not in stats:
            stats[col_name] = {'count': 0, 'mean': 0.0, 'M2': 0.0}

        st = stats[col_name]

        for val in x:
            st['count'] += 1
            delta = val - st['mean']
            st['mean'] += delta / st['count']
            delta2 = val - st['mean']
            st['M2'] += delta * delta2


    data_iter = Data_file_iterator(base_path, split='train')

    for df in data_iter:

        # --- update stats ---
        for col in df.select_dtypes(include=['number']).columns:
            update_welford(col, df[col].to_numpy())

        # --- collect samples for fitting ---
        if {'ConstellationType', 'SvElevationDegrees', 'Cn0DbHz'}.issubset(df.columns):
            mask = (~df['SvElevationDegrees'].isna()) & (~df['Cn0DbHz'].isna())
            sub = df.loc[mask, ['ConstellationType', 'SvElevationDegrees', 'Cn0DbHz']]

            # no iterrows – fully vectorised
            for constell, group in sub.groupby('ConstellationType'):
                elevs = group['SvElevationDegrees'].to_numpy(dtype=float)
                cn0s = group['Cn0DbHz'].to_numpy(dtype=float)
                samples[constell].extend(zip(elevs, cn0s))

        print(f"Processed file: {data_iter.get_current_file()}")


    # --------------------------------------------------------------
    # 2. Finalise statistics
    # --------------------------------------------------------------

    final_stats = []

    for col, s in stats.items():
        if s['count'] < 2:
            mean = s['mean']
            std_dev = 0.0
        else:
            mean = s['mean']
            variance = s['M2'] / s['count']
            std_dev = np.sqrt(variance)

        final_stats.append((col, mean, std_dev))

    if os.path.exists(output_file):
        os.remove(output_file)

    stats_df = pd.DataFrame(final_stats, columns=['column_name', 'mean', 'std_dev'])
    stats_df.to_csv(output_file, index=False)

    print("Dataset statistics saved to dataset_stats.csv")

    # for sat_key, pair_list in samples.items():
    #     arr = np.array(pair_list)
    #     elevs = arr[:, 0]
    #     cn0s = arr[:, 1]
    #     c0, c1 = coefs[sat_key]
    #     expected_vals = model_func(elevs, c0, c1)

    #     plt.figure()
    #     plt.scatter(elevs, cn0s, alpha=0.5, label='Measured')
    #     plt.scatter(elevs, expected_vals, color='red', marker='x', label='Expected')
    #     plt.title(f'C/N₀ vs Elevation for {sat_key}')
    #     plt.xlabel('Elevation (deg)')
    #     plt.ylabel('C/N₀ (dB-Hz)')
    #     plt.legend()
    #     plt.savefig(f"{sat_key}_Cn0_vs_Elevation.png")
    #     plt.close()
    
