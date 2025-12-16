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
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

SPEED_OF_LIGHT = 299792458.0
C = SPEED_OF_LIGHT

def compute_wavelength(carrier_frequency_hz):
    return C / float(carrier_frequency_hz)

def timestamps_seconds(time_nanos, full_bias_nanos, time_offset_nanos):
    t = np.array(time_nanos, dtype=np.float64)
    to = np.array(time_offset_nanos, dtype=np.float64)
    return (t - float(full_bias_nanos) + to) * 1e-9

def cn0_to_linear(cn0_db_hz):
    cn0 = np.array(cn0_db_hz, dtype=np.float64)
    return 10.0 ** (cn0 / 10.0)

def reconstruct_phase_from_adr(adr_m, carrier_frequency_hz):
    lam = compute_wavelength(carrier_frequency_hz)
    adr = np.array(adr_m, dtype=np.float64)

    delta_adr = np.diff(adr, prepend=adr[0])
    delta_phi = 2.0 * math.pi * (delta_adr / lam)

    phase_unwrapped = np.cumsum(delta_phi)
    phase_wrapped = (phase_unwrapped + math.pi) % (2.0 * math.pi) - math.pi

    return phase_unwrapped, phase_wrapped

def reconstruct_phase_from_rate(pr_rate_m_s, timestamps_s, carrier_frequency_hz):
    lam = compute_wavelength(carrier_frequency_hz)
    pr_rate = np.array(pr_rate_m_s, dtype=np.float64)

    phase_rate = -2.0 * math.pi * pr_rate / lam
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

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"

    data_iter = Data_file_iterator(base_path, split='train', limit=10)

    for df in data_iter:
        # for epoch_id, epoch_df in df.groupby('epoch_id'):
        #     print(f"Epoch ID: {epoch_id}")

                    # group by satellite
        for (constel, svid), sat_df in df.groupby(['ConstellationType', 'Svid']):

            # extract numpy arrays for convenience
            time_nanos = sat_df["TimeNanos"].values
            full_bias_nanos = sat_df["FullBiasNanos"].values[0]
            time_offset_nanos = sat_df["TimeOffsetNanos"].values
            cn0_db_hz = sat_df["Cn0DbHz"].values
            adr_m = sat_df["AccumulatedDeltaRangeMeters"].values
            pr_rate_m_s = sat_df["PseudorangeRateMetersPerSecond"].values
            carrier_frequency_hz = sat_df["CarrierFrequencyHz"].values[0]

            # timestamps in seconds
            timestamps_s = timestamps_seconds(
                time_nanos=time_nanos,
                full_bias_nanos=full_bias_nanos,
                time_offset_nanos=time_offset_nanos
            )

            # signal strength (linear, proportional to energy)
            cn0_linear = cn0_to_linear(cn0_db_hz)

            # phase from ADR
            phase_adr_unwrapped, phase_adr_wrapped = reconstruct_phase_from_adr(
                adr_m=adr_m,
                carrier_frequency_hz=carrier_frequency_hz
            )

            # phase from pseudorange rate
            phase_rate_unwrapped, phase_rate_wrapped = reconstruct_phase_from_rate(
                pr_rate_m_s=pr_rate_m_s,
                timestamps_s=timestamps_s,
                carrier_frequency_hz=carrier_frequency_hz
            )

            phase_adr_unwrapped_rate = differentiate(phase_adr_unwrapped, timestamps_s)
            phase_rate_unwrapped_rate = differentiate(phase_rate_unwrapped, timestamps_s)

            # at this point you now have:
            # timestamps_s              -> time axis
            # cn0_linear                -> signal strength over time
            # phase_adr_unwrapped_rate  -> rate of change of phase evolution from ADR
            # phase_rate_unwrapped_rate -> rate of change of phase evolution from range rate

            print(f"  SVID {svid}: samples={len(sat_df)}")