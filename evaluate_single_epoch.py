import pandas as pd
import os
import numpy as np
import torch
from internals.drive_data import Drive
from internals.evaluate import setup, run
from internals.common import compute_residual_matrix
from model import Gnss_single_epoch_net


def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor]:
    feats = epoch_df[features].to_numpy(dtype=np.float32)
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
    res_matrix = compute_residual_matrix(epoch_df)
    return feats, res_matrix

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    drive = Drive(os.path.join(base_path, "2023-05-09-21-32-us-ca-mtv-pe1", "pixel7pro"), preprocessed=True)

    print(f"Evaluating file: {drive.get_directory_name()}")
    setup(Gnss_single_epoch_net, "single_epoch_network.pt", kf=False)
    df, truth_df = drive.get_dataframes()
    save_name = os.path.basename(os.path.dirname(os.path.dirname(drive.get_directory_name()))) + "_trajectory_map_single_epoch.html"
    run(df, truth_df, get_feats, save_name)

#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
#     data_iter = Data_file_iterator(base_path, split='train', prefix='2021-07-14-20-50-us-ca-mtv-e')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-04-01-18-22-us-ca-lax-t')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-09-06-00-01-us-ca-routen')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-03-08-21-34-us-ca-mtv-u')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-05-09-21-32-us-ca-mtv-pe1')