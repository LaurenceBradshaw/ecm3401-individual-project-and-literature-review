import pandas as pd
import os
import numpy as np
import torch
from argparse import ArgumentParser
from internals.epoch_manager import Epoch_manager
from internals.train import setup, run, step_scheduler
from internals.drive_data import Drive_iterator
from internals.common import compute_residual_matrix
from model import Gnss_multi_epoch_net, get_training_drives

epoch_manager = None

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats_np = epoch_df[features].to_numpy(dtype=np.float64)
    feats = torch.tensor(feats_np, dtype=torch.float64, device=device)

    # update temporal histories
    sat_ids = epoch_df["sat_identifier"].tolist()

    for sat_id, feat_vec in zip(sat_ids, feats):
        epoch_manager.add_entry(sat_id, feat_vec.detach().cpu())

    # retrieve per-satellite sequences
    history_dict = epoch_manager.batch_history(sat_ids)

    # Build padded tensor: (num_sats, max_len, feat_dim)
    sequences = []
    lengths = []

    for sat_id in sat_ids:
        seq_list = history_dict[sat_id] # list of tensors of shape (feat_dim,)
        lengths.append(len(seq_list))

        # stack into (len_i, feat_dim)
        seq_tensor = torch.stack(seq_list, dim=0)
        sequences.append(seq_tensor)

    # Pad to max length
    max_len = max(lengths)
    feat_dim = sequences[0].shape[-1]
    num_sats = len(sequences)

    sats_tensor = torch.zeros((num_sats, max_len, feat_dim), dtype=torch.float64, device=device)

    for i, seq in enumerate(sequences):
        t = seq.shape[0]
        sats_tensor[i, :t] = seq.to(device)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    pr_res_matrix, prr_res_matrix = compute_residual_matrix(epoch_df)
    pr_res_matrix = torch.tensor(pr_res_matrix, dtype=torch.float64, device=device)
    prr_res_matrix = torch.tensor(prr_res_matrix, dtype=torch.float64, device=device)

    return sats_tensor, lengths_tensor, pr_res_matrix, prr_res_matrix

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
        default="./multi_epoch_network.pt", 
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
    setup(base_path, Gnss_multi_epoch_net, args.compute_baseline)

    num_epochs = 10

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
            epoch_manager = Epoch_manager(10)
            run(df, truth_df, get_feats, save_path=args.output_path)
            step_scheduler()