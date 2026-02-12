import pandas as pd
import os
import numpy as np
import torch
from internals.drive_data import Drive
from internals.epoch_manager import Epoch_manager
from internals.evaluate import setup, run
from internals.common import compute_residual_matrix
from model import Gnss_multi_epoch_net

epoch_manager = None

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats_np = epoch_df[features].to_numpy(dtype=np.float64)
    feats = torch.tensor(feats_np, dtype=torch.float64, device=device)

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
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    # drive = Drive(os.path.join(base_path, "2022-02-24-18-29-us-ca-lax-o", "mi8"))  # massive los and multipath
    drive = Drive(os.path.join(base_path, "2020-06-25-00-34-us-ca-mtv-sb-101", "pixel4xl"))
    # drive = Drive(os.path.join(base_path, "2023-09-06-00-01-us-ca-routen", "pixel4xl"))
    # drive = Drive(os.path.join(base_path, "2023-03-08-21-34-us-ca-mtv-u", "pixel7pro")) # test candidate (makes it look good)
    # drive = Drive(os.path.join(base_path, "2021-12-08-20-28-us-ca-lax-c", "sm-g988b")) # test candidate (makes it look good)
    # drive = Drive(os.path.join(base_path, "2021-12-08-18-52-us-ca-lax-b", "pixel6pro"))
    # drive = Drive(os.path.join(base_path, "2023-05-24-20-26-us-ca-sjc-ge2", "pixel7pro")) # train candidate
    # drive = Drive(os.path.join(base_path, "2023-05-19-20-10-us-ca-mtv-ie2", "sm-s908b")) # weirdly bad

    print(f"Evaluating file: {drive.get_directory_name()}")
    setup(Gnss_multi_epoch_net, "multi_epoch_network.pt", kf=False)
    df, truth_df = drive.get_dataframes()
    save_name = os.path.basename(os.path.dirname(drive.get_directory_name())) + "_trajectory_map_multi_epoch.html"
    epoch_manager = Epoch_manager(10)
    run(df, truth_df, get_feats, save_name)

    
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
#     data_iter = Data_file_iterator(base_path, split='train', prefix='2021-07-14-20-50-us-ca-mtv-e')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-04-01-18-22-us-ca-lax-t')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-09-06-00-01-us-ca-routen')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-03-08-21-34-us-ca-mtv-u')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-05-09-21-32-us-ca-mtv-pe1')

# 2020-06-25-00-34-us-ca-mtv-sb-101\pixel4xl   with kf shows a section where baseline diverges slightly, likely due to velocity