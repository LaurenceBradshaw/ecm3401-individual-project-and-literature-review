import numpy as np

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