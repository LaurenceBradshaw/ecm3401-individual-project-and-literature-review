import pandas as pd
import os
import numpy as np
import torch
from internals.drive_data import Drive_iterator
from internals.train import setup, run
from internals.common import compute_residual_matrix
from model import Gnss_single_epoch_net

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    feats = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
    pr_res_matrix, prr_res_matrix = compute_residual_matrix(epoch_df)
    pr_res_matrix = torch.tensor(pr_res_matrix, dtype=torch.float32, device=device)
    prr_res_matrix = torch.tensor(prr_res_matrix, dtype=torch.float32, device=device)
    return feats, pr_res_matrix, prr_res_matrix

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    drive_iter = Drive_iterator(
        # TODO: hand select these
        drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))]
    )

    setup(base_path, Gnss_single_epoch_net)

    n_drives = drive_iter.nitems()
    for i, drive in enumerate(drive_iter):
        print(f"[{i+1}/{n_drives}] Processing file: {drive.get_directory_name()}")
        df, truth_df = drive.get_dataframes()
        run(df, truth_df, get_feats, "single_epoch_network.pt")
