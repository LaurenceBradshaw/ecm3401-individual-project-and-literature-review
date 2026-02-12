import pandas as pd
import os
import numpy as np
import torch
from argparse import ArgumentParser
from internals.drive_data import Drive_iterator
from internals.train import TRAINING_DRIVES, setup, run
from internals.common import compute_residual_matrix
from model import Gnss_single_epoch_net

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    feats = epoch_df[features].to_numpy(dtype=np.float64)
    feats = torch.tensor(feats, dtype=torch.float64, device=device)
    pr_res_matrix, prr_res_matrix = compute_residual_matrix(epoch_df)
    pr_res_matrix = torch.tensor(pr_res_matrix, dtype=torch.float64, device=device)
    prr_res_matrix = torch.tensor(prr_res_matrix, dtype=torch.float64, device=device)
    return feats, pr_res_matrix, prr_res_matrix

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument(
        "-b", "--base_path", 
        type=str, 
        default="./smartphone-decimeter-2023/sdc2023/train", 
        help="Base path to training data. (default: %(default)s)"
        ) 
    parser.add_argument(
        "-o", "--output_path", 
        type=str, 
        default="./single_epoch_network.pt", 
        help="Path to save the trained model. (default: %(default)s)"
        ) 
    args = parser.parse_args()

    base_path = args.base_path
    drive_iter = Drive_iterator(
        drive_paths=[os.path.join(base_path, p) for p in TRAINING_DRIVES],
        mode="train"
    )

    setup(base_path, Gnss_single_epoch_net)

    for i, drive in enumerate(drive_iter):
        print(f"Processing file: {drive.get_directory_name()}")
        df, truth_df = drive.get_dataframes()
        run(df, truth_df, get_feats, args.output_path)
