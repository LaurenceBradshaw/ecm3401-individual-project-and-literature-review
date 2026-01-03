import pandas as pd
import os
import numpy as np
import torch
from internals.drive_data import Drive_iterator
from internals.train import setup, run
from model import Gnss_single_epoch_net

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    feats = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
    return (feats,)

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    drive_iter = Drive_iterator(
        # TODO: hand select these
        drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))],
        preprocessed=True
    )

    setup(base_path, Gnss_single_epoch_net)

    for drive in drive_iter:
        print(f"Processing file: {drive.get_directory_name()}")
        df, truth_df = drive.get_dataframes()
        run(df, truth_df, get_feats, "single_epoch_network.pt")
