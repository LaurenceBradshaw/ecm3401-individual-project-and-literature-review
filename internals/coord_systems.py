import numpy as np
import torch

class WGS84:
    A = 6378137.0               # semi-major axis [m]
    B = 6356752.314245          # semi-minor axis [m]
    F = 1 / 298.257223563       # flattening
    E2 = F * (2 - F)            # eccentricity squared
    EP2 = (A**2 - B**2) / B**2  # second eccentricity squared


def ecef_to_lla(ecef):
    x, y, z = ecef

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    # Bowring’s method for initial latitude
    theta = np.arctan2(z * WGS84.A, p * WGS84.B)
    lat = np.arctan2(
        z + WGS84.EP2 * WGS84.B * np.sin(theta)**3,
        p - WGS84.E2 * WGS84.A * np.cos(theta)**3
    )

    sin_lat = np.sin(lat)
    N = WGS84.A / np.sqrt(1 - WGS84.E2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    return np.array([np.rad2deg(lat), np.rad2deg(lon), alt])

def ecef_to_lla_torch(ecef):
    x, y, z = ecef

    lon = torch.atan2(y, x)
    p = torch.sqrt(x**2 + y**2)

    # Bowring’s method for initial latitude
    theta = torch.atan2(z * WGS84.A, p * WGS84.B)
    lat = torch.atan2(
        z + WGS84.EP2 * WGS84.B * torch.sin(theta)**3,
        p - WGS84.E2 * WGS84.A * torch.cos(theta)**3
    )

    sin_lat = torch.sin(lat)
    N = WGS84.A / torch.sqrt(1 - WGS84.E2 * sin_lat**2)
    alt = p / torch.cos(lat) - N

    return torch.stack([torch.rad2deg(lat), torch.rad2deg(lon), alt])


def lla_to_ecef(lla):
    lat, lon, alt = lla
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    N = WGS84.A / np.sqrt(1 - WGS84.E2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - WGS84.E2) + alt) * sin_lat

    return np.array([x, y, z])

def lla_to_ecef_torch(lla):
    lat, lon, alt = lla
    lat, lon = torch.deg2rad(lat), torch.deg2rad(lon)

    sin_lat, cos_lat = torch.sin(lat), torch.cos(lat)
    sin_lon, cos_lon = torch.sin(lon), torch.cos(lon)

    N = WGS84.A / torch.sqrt(1 - WGS84.E2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - WGS84.E2) + alt) * sin_lat

    return torch.stack([x, y, z])

def ecef_to_neu_rot(ref_ecef):
    ref_lla = ecef_to_lla(ref_ecef)
    lat, lon = map(np.deg2rad, ref_lla[:2])

    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon),                np.cos(lon),               0.0        ],
        [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)]
    ])
    return R


def ecef_to_ned_rot(ref_ecef):
    R = ecef_to_neu_rot(ref_ecef).copy()
    R[2, :] *= -1.0
    return R


def ecef_to_enu_rot(ref_ecef):
    ref_lla = ecef_to_lla(ref_ecef)
    lat, lon = map(np.deg2rad, ref_lla[:2])

    R = np.array([
        [-np.sin(lon),              np.cos(lon),             0.0        ],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    return R


def ecef_to_end_rot(ref_ecef):
    R = ecef_to_enu_rot(ref_ecef).copy()
    R[2, :] *= -1.0
    return R

def neu_to_ecef_rot(ref_ecef):
    return ecef_to_neu_rot(ref_ecef).T

def ned_to_ecef_rot(ref_ecef):
    return ecef_to_ned_rot(ref_ecef).T

def enu_to_ecef_rot(ref_ecef):
    return ecef_to_enu_rot(ref_ecef).T

def end_to_ecef_rot(ref_ecef):
    return ecef_to_end_rot(ref_ecef).T

def ecef_to_ned(ecef, ref_ecef):
    R = ecef_to_ned_rot(ref_ecef)
    return R @ (ecef - ref_ecef)

def ned_to_ecef(ned, ref_ecef):
    R = ned_to_ecef_rot(ref_ecef)
    return R @ ned + ref_ecef

def ecef_to_neu(ecef, ref_ecef):
    R = ecef_to_neu_rot(ref_ecef)
    return R @ (ecef - ref_ecef)

def neu_to_ecef(neu, ref_ecef):
    R = neu_to_ecef_rot(ref_ecef)
    return R @ neu + ref_ecef

def heading_speed_to_ecef(heading_deg, speed_m_s, lat_deg, lon_deg):
    # Convert everything to numpy arrays
    heading = np.atleast_1d(np.asarray(heading_deg, dtype=float))
    speed = np.atleast_1d(np.asarray(speed_m_s, dtype=float))
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))

    # Radians
    heading_rad = np.deg2rad(heading)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Precompute trig
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # East and North unit vectors
    east = np.stack([-sin_lon, cos_lon, np.zeros_like(lon_rad)], axis=1)
    north = np.stack([
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat
    ], axis=1)

    # Direction vector (broadcasted automatically)
    sin_h = np.sin(heading_rad)[:, None]
    cos_h = np.cos(heading_rad)[:, None]
    dir_ecef = sin_h * east + cos_h * north

    # Apply speed
    v_ecef = speed[:, None] * dir_ecef
    
    return v_ecef[0]

def errors_haversine(est_lla, truth_lla):
    est = np.atleast_2d(est_lla).astype(np.float64)
    truth = np.atleast_2d(truth_lla).astype(np.float64)

    lat1 = np.deg2rad(truth[:, 0])
    lon1 = np.deg2rad(truth[:, 1])
    lat2 = np.deg2rad(est[:, 0])
    lon2 = np.deg2rad(est[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0)**2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    )

    c = 2.0 * np.arcsin(np.sqrt(a))
    horizontal_error = WGS84.A * c

    vertical_error = est[:, 2] - truth[:, 2]
    error_3d = np.sqrt(horizontal_error**2 + vertical_error**2)

    # Return scalars if input was (3,)
    if est_lla.ndim == 1:
        return horizontal_error[0], vertical_error[0], error_3d[0]

    return horizontal_error, vertical_error, error_3d