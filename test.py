import numpy as np
import os
import folium
from internals.coord_systems import ecef_to_lla, heading_speed_to_ecef
from internals.drive_data import Drive
import internals.gnss_positioning as gp

L1_MIN = 1.55e9
L1_MAX = 1.61e9

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"

    drive = Drive(os.path.join(base_path, "2022-02-24-18-29-us-ca-lax-o", "mi8"))

    print(f"Evaluating file: {drive.get_directory_name()}")
    df, truth_df = drive.get_dataframes()
    df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]
    curr_pos = None
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
    df["epoch_id"] = df.groupby('utcTimeMillis').ngroup()
    N = df['epoch_id'].nunique()

    df['CorrectedPseudorange'] = df['RawPseudorangeMeters'] - df['IonosphericDelayMeters'] - df['TroposphericDelayMeters'] - df['IsrbMeters'] + df['SvClockBiasMeters']
    df['CorrectedPseudorangeRateMetersPerSecond'] = df['PseudorangeRateMetersPerSecond'] + df['SvClockDriftMetersPerSecond']

    estimated_positions = np.zeros((N, 3))
    sat_trajectories = {}

    i = 0
    for epoch_id, epoch_df in df.groupby('epoch_id'):
        # pr = (epoch_df['RawPseudorangeMeters'] - epoch_df['IonosphericDelayMeters'] - epoch_df['TroposphericDelayMeters'] - epoch_df['IsrbMeters'] + epoch_df['SvClockBiasMeters']).to_numpy()
        # prr = (epoch_df['PseudorangeRateMetersPerSecond'] + epoch_df['SvClockDriftMetersPerSecond']).to_numpy()
        pr = epoch_df['CorrectedPseudorange'].to_numpy()
        prr = epoch_df['CorrectedPseudorangeRateMetersPerSecond'].to_numpy()
        sat_pos = epoch_df[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].to_numpy()
        sat_vel = epoch_df[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Wx = 1 / epoch_df['RawPseudorangeUncertaintyMeters'].to_numpy()
        # Wv = 1 / epoch_df['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy()
        # Wx = np.diag(Wx)
        # Wv = np.diag(Wv)

        curr_pos = gp.position(pr, prr, sat_pos, sat_vel, prev_estimate=curr_pos, Wx=None, Wv=None)

        print(f"Epoch ID: {epoch_id}")
        estimated_positions[i, :] = ecef_to_lla(curr_pos["position"])

        for _, row in epoch_df.iterrows():
            key = str(row['Svid']) + "_" + str(row['ConstellationType'])

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
        location=[estimated_positions[0,0], estimated_positions[0,1]],
        zoom_start=15,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )

    folium.PolyLine(
        list(zip(estimated_positions[:,0], estimated_positions[:,1])),
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Estimated"
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

    for sv_id, path in sat_trajectories.items():
        if len(path) < 2:
            continue
        folium.CircleMarker(location=path[0], radius=3, color="purple", fill=True).add_to(m)
        folium.CircleMarker(location=path[-1], radius=3, color="black", fill=True).add_to(m)

    m.save("test_map.html")