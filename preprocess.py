import json
import pandas as pd
import os
import numpy as np
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from internals.drive_data import Drive_iterator
from internals.epoch_manager import Epoch_manager
from internals import common, gnss_positioning as gp
from internals.constants import SAT_POS_COLS, SAT_VEL_COLS, PR_COL, PRR_COL, SPEED_OF_LIGHT

# This file is wildly inefficient since it loops through the data multiple times.
# TODO: optimise later.
# TODO: Split this file up into its 3 distinct parts: general processing, dataset stats, subset selection. Then hash can be done for each part
#       Although there are transitive dependencies. i.e., if general processing changes, then subsequent steps need to be redone. this is the only case.

L1_MIN = 1.55e9
L1_MAX = 1.61e9

CONSTELLATION_MAP = {
    0: 'UNK',
    1: 'G',
    2: 'SBAS',
    3: 'GLO',
    4: 'QZ',
    5: 'BD',
    6: 'GAL',
    7: 'I'
}

def hash_current_file() -> str:
    file_path = Path(__file__).resolve()
    hasher = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def validate_satellite_data(df: pd.DataFrame, sat_identifier: str, sat_df: pd.DataFrame) -> None:
    # Pretty arbitrary thresholds. Very generous to avoid removing actual fluctuations in the data
    MAX_SAT_JUMP_KM = 200.0
    MAX_SAT_SPEED_MPS = 18000.0
    MAX_PR_DELTA = 10000.0       # metres
    MAX_PRR_MAG = 30000.0        # m/s
    MAX_PRR_DELTA = 800.0        # m/s change per epoch

    g = sat_df.sort_values('utcTimeMillis').copy()
    idx = g.index.to_numpy()

    r = g[SAT_POS_COLS].to_numpy()
    v = g[SAT_VEL_COLS].to_numpy()
    t = g['utcTimeMillis'].to_numpy()

    # compute position deltas
    dr = np.diff(r, axis=0)
    dt = np.diff(t).astype(float) / 1000.0 
    dt[dt <= 0] = np.nan

    jump_dist = np.linalg.norm(dr, axis=1) / 1000.0
    v_est = dr / dt[:, None]
    v_est_mag = np.linalg.norm(v_est, axis=1)

    # reported velocity magnitude
    v_reported_mag = np.linalg.norm(v, axis=1)

    # flag bad epochs (1: end of delta, 0: nothing)
    bad = np.zeros(len(g), dtype=bool)
    bad[1:] |= (jump_dist > MAX_SAT_JUMP_KM)
    bad[1:] |= (v_est_mag > MAX_SAT_SPEED_MPS)
    bad |= (v_reported_mag > MAX_SAT_SPEED_MPS)

    # record initial flags
    df.loc[idx[bad], 'motion_flag'] = 'bad_motion'
    df.loc[idx[~bad], 'motion_flag'] = 'good_motion'

    pr = g[PR_COL].to_numpy()
    prr = g[PRR_COL].to_numpy()

    pr_jump = np.abs(np.diff(pr))
    pr_rate = pr_jump / dt
    prr_jump = np.abs(np.diff(prr))
    prr_rate = prr_jump / dt

    pr_bad = np.zeros(len(pr), dtype=bool)
    pr_bad[1:] |= pr_rate > MAX_PR_DELTA

    prr_bad = np.zeros(len(prr), dtype=bool)
    prr_bad[1:] |= prr_rate > MAX_PRR_DELTA
    prr_bad |= np.abs(prr) > MAX_PRR_MAG

    combined_bad = pr_bad | prr_bad

    df.loc[idx[pr_bad], 'pr_flag'] = 'bad_pr'
    df.loc[idx[~pr_bad], 'pr_flag'] = 'good_pr'
    df.loc[idx[prr_bad], 'prr_flag'] = 'bad_prr'
    df.loc[idx[~prr_bad], 'prr_flag'] = 'good_prr'
    df.loc[idx[combined_bad], 'sat_flag'] = 'bad_satellite'
    df.loc[idx[~combined_bad], 'sat_flag'] = 'good_satellite'
    # if any bad points exist, interpolate them
    if bad.any() or combined_bad.any():
        print(f"Found {np.sum(bad)} bad motion instances for satellite {sat_identifier}")
        print(f"Found {np.sum(combined_bad)} bad pr/prr measurements instances for satellite {sat_identifier}")
        print(f"Repairing only satellite position/velocity for satellites with no pr/prr issues - those will be dropped.")
        good_mask = ~bad
        good_times = t[good_mask]

        # positions
        for i, col in enumerate(SAT_POS_COLS):
            good_vals = r[good_mask, i]
            interp_vals = np.interp(t, good_times, good_vals)
            df.loc[idx, col] = interp_vals

        # velocities
        for i, col in enumerate(SAT_VEL_COLS):
            good_vals = v[good_mask, i]
            interp_vals = np.interp(t, good_times, good_vals)
            df.loc[idx, col] = interp_vals


def reconstruct_phase_from_adr(adr_m, wavelength):
    adr = np.array(adr_m, dtype=np.float64)

    delta_adr = np.diff(adr, prepend=adr[0])
    delta_phi = 2.0 * math.pi * (delta_adr / wavelength)

    phase_unwrapped = np.cumsum(delta_phi)
    phase_wrapped = (phase_unwrapped + math.pi) % (2.0 * math.pi) - math.pi

    return phase_unwrapped, phase_wrapped

def reconstruct_phase_from_rate(pr_rate_m_s, timestamps_s, wavelength):
    pr_rate = np.array(pr_rate_m_s, dtype=np.float64)

    phase_rate = -2.0 * math.pi * pr_rate / wavelength
    dt = np.diff(timestamps_s, prepend=timestamps_s[0])

    phase_unwrapped = np.cumsum(phase_rate * dt)
    phase_wrapped = (phase_unwrapped + math.pi) % (2.0 * math.pi) - math.pi

    return phase_unwrapped, phase_wrapped

def differentiate(phi, timestamps_s):
    # robust derivative for irregular timestamps (returns rad/s)
    ts = np.array(timestamps_s, dtype=np.float64)
    phi = np.array(phi, dtype=np.float64)
    dt = np.diff(ts, prepend=ts[0])
    # protect against zero dt (leave derivative zero)
    dt[dt == 0] = np.nan
    dphi = np.diff(phi, prepend=phi[0])
    dphi_dt = dphi / dt
    # replace nan with zero
    dphi_dt = np.nan_to_num(dphi_dt)
    return dphi_dt

def reconstruct_phase_data(df: pd.DataFrame, sat_df: pd.DataFrame) -> None:
    time_nanos = sat_df["TimeNanos"].to_numpy()
    full_bias_nanos = sat_df["FullBiasNanos"].to_numpy()[0]
    time_offset_nanos = sat_df["TimeOffsetNanos"].to_numpy()
    adr_m = sat_df["AccumulatedDeltaRangeMeters"].to_numpy()
    pr_rate_m_s = sat_df["PseudorangeRateMetersPerSecond"].to_numpy()
    wavelength = sat_df["wavelength"].to_numpy()[0]

    # timestamps in seconds
    timestamps_s = (time_nanos - float(full_bias_nanos) + time_offset_nanos) * 1e-9

    # phase from ADR
    phase_adr_unwrapped, phase_adr_wrapped = reconstruct_phase_from_adr(
        adr_m=adr_m,
        wavelength=wavelength
    )

    # phase from pseudorange rate
    phase_rate_unwrapped, phase_rate_wrapped = reconstruct_phase_from_rate(
        pr_rate_m_s=pr_rate_m_s,
        timestamps_s=timestamps_s,
        wavelength=wavelength
    )

    phase_adr_unwrapped_rate = differentiate(phase_adr_unwrapped, timestamps_s)
    phase_rate_unwrapped_rate = differentiate(phase_rate_unwrapped, timestamps_s)

    df.loc[sat_df.index, 'phase_adr_unwrapped_rate'] = phase_adr_unwrapped_rate
    df.loc[sat_df.index, 'phase_rate_unwrapped_rate'] = phase_rate_unwrapped_rate
    df.loc[sat_df.index, 'phase_adr_wrapped'] = phase_adr_wrapped
    df.loc[sat_df.index, 'phase_rate_wrapped'] = phase_rate_wrapped
    df.loc[sat_df.index, 'sin_phase_adr_wrapped'] = np.sin(phase_adr_wrapped)
    df.loc[sat_df.index, 'cos_phase_adr_wrapped'] = np.cos(phase_adr_wrapped)
    df.loc[sat_df.index, 'sin_phase_rate_wrapped'] = np.sin(phase_rate_wrapped)
    df.loc[sat_df.index, 'cos_phase_rate_wrapped'] = np.cos(phase_rate_wrapped)

def map_adr_state(state_int):
    """
    Convert ADR state bits to state.
    State map comes from accumulated_delta_range_state_bit_map.json in the dataset.

    Args:
        state_int: Integer representing the state bitmask

    Returns:
        dict: ADR state flags
    """
    adr_state = {
        'valid': False,
        'reset': False,
        'cycle_slip': False,
        'half_cycle_resolved': False,
        'half_cycle_reported': False
    }
    if state_int & (1 << 0):
        adr_state['valid'] = True
    if state_int & (1 << 1):
        adr_state['reset'] = True
    if state_int & (1 << 2):
        adr_state['cycle_slip'] = True
    if state_int & (1 << 3):
        adr_state['half_cycle_resolved'] = True
    if state_int & (1 << 4):
        adr_state['half_cycle_reported'] = True

    return adr_state

def map_to_generic_state(state_int):
    """
    Convert constellation-specific state bits to generic tracking states.
    State map comes from raw_state_bit_map.json in the dataset.
    
    Args:
        state_int: Integer representing the state bitmask
        
    Returns:
        dict: Generic state flags
    """
    # Initialize all to False
    generic_state = {
        'code_lock': False,
        'bit_sync': False,
        'frame_sync': False,
        'time_decoded': False,
        'ambiguity_resolved': False
    }
    
    # Code Lock (Bit 0)
    if state_int & (1 << 0):
        generic_state['code_lock'] = True
    
    # Bit/Symbol Sync (Bits 1, 5, 8, 10, 11)
    if (state_int & (1 << 1)) or (state_int & (1 << 5)) or \
       (state_int & (1 << 8)) or (state_int & (1 << 10)) or (state_int & (1 << 11)):
        generic_state['bit_sync'] = True
    
    # Frame/Subframe/Page Sync (Bits 2, 6, 9, 12, 13)
    if (state_int & (1 << 2)) or (state_int & (1 << 6)) or \
       (state_int & (1 << 9)) or (state_int & (1 << 12)) or (state_int & (1 << 13)):
        generic_state['frame_sync'] = True
    
    # Time Decoded/Known (Bits 3, 7, 14, 15)
    if (state_int & (1 << 3)) or (state_int & (1 << 7)) or \
       (state_int & (1 << 14)) or (state_int & (1 << 15)):
        generic_state['time_decoded'] = True
    
    # Ambiguity Resolved (Bit 4)
    if state_int & (1 << 4):
        generic_state['ambiguity_resolved'] = True
    
    return generic_state

def calc_satellite_data(df: pd.DataFrame, sat_id: int, sat_row: pd.Series, epoch_manager: Epoch_manager) -> None:
    # Cn0 model
    cn0 = sat_row['Cn0DbHz']
    el = sat_row['sin_elevation']
    cn0_over_sine = cn0 / np.maximum(el, 0.1) # Avoid division by zero
    df.loc[sat_id, 'cn0_over_sine'] = cn0_over_sine

    # Generic state mapping
    state = sat_row['State']
    generic_state = map_to_generic_state(state)
    for key, value in generic_state.items():
        df.loc[sat_id, key] = float(value)

    adr_state = sat_row['AccumulatedDeltaRangeState']
    mapped_adr_state = map_adr_state(adr_state)
    for key, value in mapped_adr_state.items():
        df.loc[sat_id, key] = float(value)

    # Stability metrics
    sat_identifier = sat_row['sat_identifier']
    pr = sat_row[PR_COL]
    prr = sat_row[PRR_COL]
    adr = sat_row['AccumulatedDeltaRangeMeters']

    epoch_manager.add_entry(sat_identifier, {'cn0': cn0, 'pr': pr, 'prr': prr, 'adr': adr})
    epoch_window = epoch_manager.get_history(sat_identifier)

    keys = epoch_window[0].keys()
    epoch_window = {k: [d[k] for d in epoch_window] for k in keys}

    cn0_diff = np.diff(epoch_window['cn0'], prepend=0)
    cn0_std = np.std(cn0_diff)
    cn0_stability = 1 / max(cn0_std, 1e-3) # avoid div by zero
    df.loc[sat_id, 'cn0_stability'] = cn0_stability

    pr_std = np.std(epoch_window['pr'])
    pr_stability = 1 / max(pr_std, 1e-3)
    df.loc[sat_id, 'pr_stability'] = pr_stability
    prr_std = np.std(epoch_window['prr'])
    prr_stability = 1 / max(prr_std, 1e-3)
    df.loc[sat_id, 'prr_stability'] = prr_stability

    carrier_avg = np.mean(epoch_window['adr'])
    code_carrier_div = pr - carrier_avg
    cc_div_std = np.std(code_carrier_div)
    df.loc[sat_id, 'code_carrier_div_std'] = cc_div_std

def compute_prr_baseline_weight(sat_samples: dict, row: pd.Series) -> float:
    sat_key = (row['ConstellationType'], row['Svid'])
    res = row['doppler_residual']
    b = row['Cn0DbHz'] 
    
    # Initialise per-satellite list if needed
    if sat_key not in sat_samples:
        sat_samples[sat_key] = []
    
    # Add current residual to satellite history
    sat_samples[sat_key].append(res)
    
    # Use last N samples for batch variance (or all if fewer than N)
    samples = sat_samples[sat_key][-50:] # e.g., last 50 residuals
    N = len(samples)
    
    if N > 1:
        s2 = np.var(samples, ddof=1) # unbiased sample variance
    else:
        s2 = 1e-3 # default small value for new satellite
    
    # Compute mean geometry and mean noise scale for the batch
    sin2a = 1/row['sin_elevation']**2
    c = 10**(-b/10)
    
    # Estimate k using batch residual formula
    k_hat = max(s2 - sin2a * c, 1e-6) # clamp to positive
    
    # Compute sigma^2 for weight
    sigma2 = sin2a + k_hat * c
    return 1.0 / sigma2

def calc_epoch_level_stats(df: pd.DataFrame, epoch_df: pd.DataFrame, curr_pos: dict) -> None:
    # elevation rank for all satellites in this epoch
    elevation_rank = epoch_df['SvElevationDegrees'].rank(ascending=False).to_numpy()
    df.loc[epoch_df.index, 'elevation_rank'] = elevation_rank

    # Get pseudoranges and satellite positions
    pr = epoch_df[PR_COL].to_numpy()
    prr = epoch_df[PRR_COL].to_numpy()
    sat_pos = epoch_df[SAT_POS_COLS].to_numpy()
    sat_vel = epoch_df[SAT_VEL_COLS].to_numpy()

    # Compute receiver position using least squares with no weighting
    curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos)

    # Compute residual for all included satellites
    x = np.zeros(4)
    x[:3] = curr_pos["position"]
    x[3] = curr_pos["clock_bias"]
    res = gp.pr_residuals(x, sat_pos, pr)
    df.loc[epoch_df.index, 'residual'] = res

    rms_residual = np.sqrt(np.mean(res**2))
    df.loc[epoch_df.index, 'rms_residual'] = rms_residual

    v = np.zeros(4)
    v[:3] = curr_pos['velocity']
    v[3] = curr_pos['clock_drift']
    rate_res = gp.prr_residuals(v, sat_vel, prr, x, sat_pos)
    df.loc[epoch_df.index, 'rate_residual'] = rate_res

    los = gp.los_vector(curr_pos['position'], sat_pos)[0]
    wavelength = epoch_df['wavelength'].to_numpy()
    rho_geom = np.sum(los * (sat_vel - curr_pos['velocity']), axis=1)
    rho_clk = curr_pos['clock_drift'] - epoch_df["SvClockDriftMetersPerSecond"].to_numpy()
    rho_pred = rho_geom + rho_clk
    D_pred = -(rho_pred / wavelength)
    D_meas = -prr / wavelength
    D_res = D_meas - D_pred

    df.loc[epoch_df.index, 'doppler_residual'] = D_res
    epoch_df.loc[epoch_df.index, 'doppler_residual'] = D_res
    

def create_dataset_stats(base_path: str):
    stats = {} # column -> {count, mean, M2}
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
    
    drive_iter = Drive_iterator(
        drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))],
    )

    for drive in drive_iter:
        df, _ = drive.get_dataframes()
        # update stats
        for col in df.select_dtypes(include=['number']).columns:
            update_welford(col, df[col].to_numpy())

        # collect samples for fitting
        if {'ConstellationType', 'SvElevationDegrees', 'Cn0DbHz'}.issubset(df.columns):
            mask = (~df['SvElevationDegrees'].isna()) & (~df['Cn0DbHz'].isna())
            sub = df.loc[mask, ['ConstellationType', 'SvElevationDegrees', 'Cn0DbHz']]

            for constell, sat_df in sub.groupby('ConstellationType'):
                elevs = sat_df['SvElevationDegrees'].to_numpy(dtype=float)
                cn0s = sat_df['Cn0DbHz'].to_numpy(dtype=float)
                samples[constell].extend(zip(elevs, cn0s))

        print(f"Processed file: {drive.get_directory_name()}")

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

def adr_is_usable(adr_state):
    return (
        adr_state['valid']
        and not adr_state['reset']
        and not adr_state['cycle_slip']
    )

if __name__ == "__main__":
    # Hash of this python file
    hash = hash_current_file()
    if not os.path.exists("preprocess_hash_table.json"):
        hash_table = {}
    else:        
        with open("preprocess_hash_table.json", "r") as f:
            hash_table = json.load(f)

    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    output_file = f"{os.path.dirname(base_path)}/dataset_stats.csv"
    # data_iter = Data_file_iterator(base_path, split='train', preprocessed=False)
    
    drive_iter = Drive_iterator(
        drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))],
        preprocessed=False
    )

    n_drives = drive_iter.nitems()
    drive_i = 0
    # TODO: parallelise this processing (will have to log to separate file or something)
    for drive in drive_iter:
        if drive.get_directory_name() in hash_table and hash_table[drive.get_directory_name()] == hash:
            print(f"[{drive_i+1}/{n_drives}] Skipping file (already processed with same code): {drive.get_directory_name()}")
            drive_i += 1
            continue
        print(f"[{drive_i+1}/{n_drives}] Pre-processing file: {drive.get_directory_name()}")
        drive_i += 1
        df, truth_df = drive.get_dataframes()
        
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
            'Cn0DbHz',
            'AccumulatedDeltaRangeMeters'
        ])

        if df.empty: # No valid data after filtering
            print(f"No valid data after filtering, skipping drive: {drive.get_directory_name()}")
            continue

        # Apply corrections to pseudorange and pseudorange rate
        df[PR_COL] = df['RawPseudorangeMeters'] - df['IonosphericDelayMeters'] - df['TroposphericDelayMeters'] - df['IsrbMeters'] + df['SvClockBiasMeters']
        df[PRR_COL] = df['PseudorangeRateMetersPerSecond'] + df['SvClockDriftMetersPerSecond']

        # Derived features
        df["sin_elevation"] = np.sin(np.deg2rad(df['SvElevationDegrees']))
        df["cos_elevation"] = np.cos(np.deg2rad(df['SvElevationDegrees']))
        df["sin_azimuth"] = np.sin(np.deg2rad(df['SvAzimuthDegrees']))
        df["cos_azimuth"] = np.cos(np.deg2rad(df['SvAzimuthDegrees']))
        df["wavelength"] = SPEED_OF_LIGHT / df['CarrierFrequencyHz']
        df['Cn0DbHz_linear'] = 10 ** (df['Cn0DbHz'] / 10)

        # Identify if satellite was seen in previous epoch or is newly appeared
        df['prev_epoch_seen'] = df.groupby(['ConstellationType', 'Svid'])['epoch_id'].shift()
        df["seen_t_minus_1"] = df["epoch_id"] - df["prev_epoch_seen"] == 1

        df["sat_identifier"] = (
            df["ConstellationType"].map(lambda c: CONSTELLATION_MAP.get(c, "UNK"))
            + "_" 
            + df["Svid"].astype(int).astype(str)
        )
        
        # Process each satellite individually for all epochs
        for (constellation, svid), sat_df in df.groupby(['ConstellationType', 'Svid']):
            validate_satellite_data(df, sat_df['sat_identifier'].values[0], sat_df)
            reconstruct_phase_data(df, sat_df)

        # Remove rows marked as bad_satellite, since interpolation on the sat pos/vel is not sufficient
        df = df[df['sat_flag'] != 'bad_satellite']
        # Filter out epochs with insufficient satellites
        # Should be done with removing rows now
        df = df.groupby('epoch_id').filter(lambda x: len(x) >= 5)
        if df.empty:
            print(f"No valid data after filtering, skipping drive: {drive.get_directory_name()}")
            continue

        # Check to see if any duplicate satellites in the same epoch and try to resolve them
        duplicate_epochs = df.duplicated(subset=['epoch_id', 'ConstellationType', 'Svid', 'SignalType'], keep=False)
        if duplicate_epochs.any():
            print(f"Found duplicate satellite entries in the same epoch. Attempting to resolve...")
        duplicate_rows = df[duplicate_epochs]
        for epoch_id, dup_epoch_df in duplicate_rows.groupby('epoch_id'):
            for (constellation, svid, signal_type), group in dup_epoch_df.groupby(['ConstellationType', 'Svid', 'SignalType']):
                if len(group) <= 1:
                    assert False, "Should only be processing duplicates here."
                times = group['utcTimeMillis'].to_numpy()
                time_diffs = np.diff(np.sort(times))
                if np.all(time_diffs == 0):
                    # All timestamps are the same, drop one of the duplicates
                    df.drop(group.index[1:], inplace=True)
                else:
                    # Push duplicates to next epoch if possible
                    # Not possible if next epoch already has that satellite
                    next_epoch_id = epoch_id + 1
                    next_epoch_df = df[df['epoch_id'] == next_epoch_id]
                    if not ((next_epoch_df['ConstellationType'] == constellation) & (next_epoch_df['Svid'] == svid) & (next_epoch_df['SignalType'] == signal_type)).any():
                        # Push to next epoch
                        df.loc[group.index, 'epoch_id'] = next_epoch_id
                    else:
                        # Drop duplicates
                        df.drop(group.index[1:], inplace=True)
                        
        # Assert no more duplicates
        duplicate_epochs = df.duplicated(subset=['epoch_id', 'ConstellationType', 'Svid', 'SignalType'], keep=False)
        assert not duplicate_epochs.any(), "Duplicate satellite entries still exist after resolution."
        
        curr_pos = None
        curr_pos_weighted = None
        sat_samples = {}
        epoch_manager = Epoch_manager(max_history=10)
        # Process each epoch individually
        print("Calculating epoch-level statistics and weights...")
        for epoch_id, epoch_df in df.groupby('epoch_id'):
            calc_epoch_level_stats(df, epoch_df, curr_pos)

            # Baseline weights
            pr_weights_baseline = 1 / ((1/epoch_df['sin_elevation'])**2 + (4*10**(-epoch_df['Cn0DbHz']/30))**2)
            df.loc[epoch_df.index, 'pr_baseline_weight'] = pr_weights_baseline
            prr_weights_baseline = np.zeros(len(epoch_df))
            i = 0

            # For each satellite in epoch
            for sat_id, sat_row in epoch_df.iterrows():
                calc_satellite_data(df, sat_id, sat_row, epoch_manager)
                prr_weights_baseline[i] = compute_prr_baseline_weight(sat_samples, sat_row)
                i += 1
                
                
            # Save weights back to DataFrame
            df.loc[epoch_df.index, 'prr_baseline_weight'] = prr_weights_baseline

        prev_epoch = None
        df['adr_td_residual'] = 0.0
        df['adr_td_valid'] = 0.0
        print("Calculating ADR time-differenced residuals...")
        for epoch_id, epoch_df in df.groupby('epoch_id'):
            if prev_epoch is not None:
                common_sats = set(epoch_df['sat_identifier']).intersection(set(prev_epoch['sat_identifier']))
                for sat_identifier in common_sats:
                    curr_sat_idx = epoch_df[epoch_df['sat_identifier'] == sat_identifier].index[0]
                    prev_sat_idx = prev_epoch[prev_epoch['sat_identifier'] == sat_identifier].index[0]
                    curr_state = epoch_df.loc[curr_sat_idx, ['valid', 'reset', 'cycle_slip']].astype(bool)
                    prev_state = prev_epoch.loc[prev_sat_idx, ['valid', 'reset', 'cycle_slip']].astype(bool)
                    if not (adr_is_usable(curr_state) and adr_is_usable(prev_state)):
                        df.loc[curr_sat_idx, 'adr_td_residual'] = 0.0
                        df.loc[curr_sat_idx, 'adr_td_valid'] = 0.0
                        continue
                    
                    curr_adr = epoch_df.loc[curr_sat_idx, 'AccumulatedDeltaRangeMeters']
                    prev_adr = prev_epoch.loc[prev_sat_idx, 'AccumulatedDeltaRangeMeters']
                    curr_pr_rate = epoch_df.loc[curr_sat_idx, 'PseudorangeRateMetersPerSecond']

                    dt = (epoch_df.loc[curr_sat_idx, 'utcTimeMillis'] - prev_epoch.loc[prev_sat_idx, 'utcTimeMillis']) / 1000.0

                    delta_adr = curr_adr - prev_adr
                    predicted_delta = curr_pr_rate * dt

                    residual = delta_adr - predicted_delta

                    df.loc[curr_sat_idx, 'adr_td_residual'] = residual
                    df.loc[curr_sat_idx, 'adr_td_valid'] = 1.0
            prev_epoch = epoch_df

        df.to_csv(os.path.join(f"{drive.get_directory_name()}", "device_gnss_preprocessed.csv"), index=False)
        # Update hash table
        hash_table[drive.get_directory_name()] = hash

        with open("preprocess_hash_table.json", "w") as f:
            json.dump(hash_table, f)

    print("Pre-processing complete. Computing dataset statistics...")
    create_dataset_stats(base_path)

    # For each file, find the largest area of pseudorange residuals (from truth), and save as a new file with the 100 surrounding epochs.
    # Can repeat n times per file.
    window_size = 100          # Total number of epochs per window
    half_window = window_size // 2
    n_parts = 3                # Max number of windows per drive

    drive_iter = Drive_iterator(
        drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))],
    )
    print("Creating focused subsets of data based on pseudorange residuals...")
    drive_i = 0
    n_drives = drive_iter.nitems()
    for drive in drive_iter:
        if f"parts_{drive.get_directory_name()}" in hash_table and hash_table[f"parts_{drive.get_directory_name()}"] == hash:
            print(f"[{drive_i+1}/{n_drives}] Skipping drive for focused subsets (already processed with same code): {drive.get_directory_name()}")
            drive_i += 1
            continue
        print(f"[{drive_i+1}/{n_drives}] Processing drive for focused subsets: {drive.get_directory_name()}")
        drive_i += 1
        df, truth_df = drive.get_dataframes()
        
        residuals = []
        epoch_ids = []

        # Compute residuals per epoch
        for i, epoch_df in df.groupby('epoch_id'):
            pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)

            pr = epoch_df[PR_COL].to_numpy()
            prr = epoch_df[PRR_COL].to_numpy()
            sat_pos = epoch_df[SAT_POS_COLS].to_numpy()
            sat_vel = epoch_df[SAT_VEL_COLS].to_numpy()

            x = np.zeros(4, dtype=np.float64)
            x[0:3] = pos_truth.numpy().flatten()
            x[3], _ = gp.estimate_rx_clock_bias_and_drift(pr, prr, sat_pos, sat_vel,
                                                            pos_truth.numpy().flatten(),
                                                            vel_truth.numpy().flatten())

            residual = gp.pr_residuals(x, sat_pos, pr)
            residuals.append(residual)
            epoch_ids.append(i)

        residuals_array = np.array([np.linalg.norm(r) for r in residuals])
        used_mask = np.zeros_like(residuals_array, dtype=bool) # Track epochs already in windows

        for part_idx in range(1, n_parts + 1):
            # Mask already used epochs
            masked_residuals = np.where(used_mask, -1, residuals_array)
            if masked_residuals.size <= 0:
                break
            if masked_residuals.max() < 0:
                break # No remaining residuals

            idx_max = masked_residuals.argmax()

            # Determine window boundaries
            start = max(0, idx_max - half_window)
            end = min(len(residuals), idx_max + half_window)

            # Mark these epochs as used
            used_mask[start:end] = True

            selected_epochs = epoch_ids[start:end]

            df_to_save = df[df['epoch_id'].isin(selected_epochs)].copy()

            out_path = os.path.join(drive.get_directory_name(), f"device_gnss_preprocessed_part{part_idx}.csv")
            df_to_save.to_csv(out_path, index=False)

        # Update hash table
        hash_table[f"parts_{drive.get_directory_name()}"] = hash

        with open("preprocess_hash_table.json", "w") as f:
            json.dump(hash_table, f)

        



    
    
