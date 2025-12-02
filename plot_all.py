from data_file_iter import Data_file_iterator
import numpy as np
import pandas as pd
from utils import gnss_positioning as gp
from utils.coord_systems import ecef_to_lla, lla_to_ecef
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

L1_MIN = 1.55e9
L1_MAX = 1.61e9

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"
    data_iter = Data_file_iterator(base_path, split='train', limit=750)

    m = folium.Map(
        location=[0, 0],
        zoom_start=15,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )

    for df in data_iter:
        print(f"Processing file: {data_iter.get_current_file()}")

        first_time_stamp = df['utcTimeMillis'].iloc[0]
        df['epoch_id'] = ((df['utcTimeMillis'] - first_time_stamp) / 1000).round().astype(int)
        df = df[(df['CarrierFrequencyHz'] >= L1_MIN) & (df['CarrierFrequencyHz'] <= L1_MAX)]

        # Only epochs with enough satellites
        df = df.groupby('epoch_id').filter(lambda x: len(x) >= 5)
        df = df.dropna(subset=[
            'RawPseudorangeMeters', 'PseudorangeRateMetersPerSecond',
            'SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters',
            'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond',
            'SvElevationDegrees', 'SvAzimuthDegrees', 'Cn0DbHz'
        ])

        truth_df = data_iter.get_truth_file()
        truth_df['epoch_id'] = ((truth_df['UnixTimeMillis'] - first_time_stamp) / 1000).round().astype(int)

        # Compute proportion of satellites reporting multipath per epoch
        grouped = df.groupby('epoch_id')
        multipath_prop = grouped['MultipathIndicator'].sum() / grouped['MultipathIndicator'].count()
        multipath_prop = multipath_prop.reindex(truth_df['epoch_id']).fillna(0)

        # Colour map: 0.0 → green, 1.0 → red
        colormap = cm.linear.viridis.scale(0, 1)

        # Prepare coordinates
        coords = list(zip(truth_df['LatitudeDegrees'], truth_df['LongitudeDegrees']))
        props = multipath_prop.to_numpy()

        # Draw line segments with proportional colours
        for i in range(len(coords) - 1):
            c1, c2 = coords[i], coords[i + 1]
            intensity = props[i]  # proportion 0–1
            colour = colormap(intensity)

            folium.PolyLine(
                locations=[c1, c2],
                color=colour,
                weight=4,
                opacity=0.5,
                tooltip=f"From {data_iter.get_current_file()}"
            ).add_to(m)

    colormap.caption = "Proportion of Satellites Reporting Multipath"
    colormap.add_to(m)
    m.save("all_trajectories.html")