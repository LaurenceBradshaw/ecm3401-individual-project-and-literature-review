import pandas as pd
import os
import numpy as np
import torch
from internals.epoch_manager import Epoch_manager
from internals.train import setup, run
from internals.drive_data import Drive_iterator
from internals.common import compute_residual_matrix
from model import Gnss_multi_epoch_net

epoch_manager = None

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats_np = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)

    # ----- 1) update temporal histories -----
    sat_ids = epoch_df["sat_identifier"].tolist()

    for sat_id, feat_vec in zip(sat_ids, feats):
        epoch_manager.add_entry(sat_id, feat_vec.detach().cpu())

    # ----- 2) retrieve per-satellite sequences -----
    history_dict = epoch_manager.batch_history(sat_ids)

    # Build padded tensor: (num_sats, max_len, feat_dim)
    sequences = []
    lengths = []

    for sat_id in sat_ids:
        seq_list = history_dict[sat_id]     # list of tensors of shape (feat_dim,)
        lengths.append(len(seq_list))

        # stack into (len_i, feat_dim)
        seq_tensor = torch.stack(seq_list, dim=0)
        sequences.append(seq_tensor)

    # Pad to max length
    max_len = max(lengths)
    feat_dim = sequences[0].shape[-1]
    num_sats = len(sequences)

    sats_tensor = torch.zeros((num_sats, max_len, feat_dim), dtype=torch.float32, device=device)

    for i, seq in enumerate(sequences):
        t = seq.shape[0]
        sats_tensor[i, :t] = seq.to(device)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    pr_res_matrix, prr_res_matrix = compute_residual_matrix(epoch_df)
    pr_res_matrix = torch.tensor(pr_res_matrix, dtype=torch.float32, device=device)
    prr_res_matrix = torch.tensor(prr_res_matrix, dtype=torch.float32, device=device)

    return sats_tensor, lengths_tensor, pr_res_matrix, prr_res_matrix

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    drive_iter = Drive_iterator(
        # TODO: hand select these
        # drive_paths=[os.path.join(base_path, d, p) for d in os.listdir(base_path) for p in os.listdir(os.path.join(base_path, d))],
        drive_paths=[os.path.join(base_path, "2020-06-25-00-34-us-ca-mtv-sb-101", "pixel4xl"), 
                     os.path.join(base_path, "2022-02-24-18-29-us-ca-lax-o", "mi8"),
                     os.path.join(base_path, "2023-09-06-00-01-us-ca-routen", "pixel4xl"),
                     os.path.join(base_path, "2023-03-08-21-34-us-ca-mtv-u", "pixel7pro")
                     ]
    )

    setup(base_path, Gnss_multi_epoch_net)

    n_drives = drive_iter.nitems()
    for i, drive in enumerate(drive_iter):
        print(f"[{i+1}/{n_drives}] Processing file: {drive.get_directory_name()}")
        df, truth_df = drive.get_dataframes()
        epoch_manager = Epoch_manager()
        run(df, truth_df, get_feats, save_path="multi_epoch_network.pt")