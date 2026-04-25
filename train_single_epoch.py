import pandas as pd
import os
import numpy as np
import torch
from argparse import ArgumentParser
from internals.drive_data import Drive_iterator
from internals.train import setup, run, step_scheduler
from internals.common import compute_residual_matrix
from model import Gnss_single_epoch_net, get_training_drives

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    """
    Get the features for the current epoch.

    Parameters
    ----------
    epoch_df: pd.DataFrame
        The DataFrame containing the data for the current epoch.
    features: list[str]
        The list of feature names to extract.
    device: str
        The device on which to place the tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the tensor of features and the tensors of residual matrices.
    """
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
    parser.add_argument(
        "-c", "--compute_baseline",
        default=False,
        action="store_true",
        help="Whether to compute baseline performance for comparison. (default: %(default)s)"
    )
    args = parser.parse_args()

    base_path = args.base_path
    setup(base_path, Gnss_single_epoch_net, args.compute_baseline)

    num_epochs = 5

    for epoch_num in range(1, num_epochs + 1):
        print(f"======== Starting training for epoch {epoch_num} / {num_epochs} ========")
        drive_iter = Drive_iterator(
            drive_paths=get_training_drives(base_path),
            mode="train"
        )

        for i, drive in enumerate(drive_iter):
            print(f"Epoch: {epoch_num} / {num_epochs}")
            print(f"Processing file: {drive.get_directory_name()}")
            df, truth_df = drive.get_dataframes()
            run(df, truth_df, get_feats, args.output_path)
            
        step_scheduler()
