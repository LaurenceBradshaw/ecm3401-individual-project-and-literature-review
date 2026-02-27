import numpy as np
import pandas as pd
import os
from collections import defaultdict
from internals.drive_data import Drive_iterator
from internals.preprocessing.hash_file import hash_file

HASH_FILE = "dataset_stats_hash.json"

def create_dataset_stats(base_path: str, output_file: str, hash_table: dict, per_file_hash: str) -> None:
    hash = hash_file(__file__)
    joint_hash = hash + per_file_hash
    if "dataset_stats" in hash_table and hash_table["dataset_stats"] == joint_hash:
        print("Dataset statistics are up to date, skipping creation.")
        return
    
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
        drive_paths=[
            os.path.join(base_path, d, p)
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
            for p in os.listdir(os.path.join(base_path, d))
            if os.path.isdir(os.path.join(base_path, d, p))
        ],
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

    hash_table["dataset_stats"] = joint_hash