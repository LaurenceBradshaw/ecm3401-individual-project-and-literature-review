import pandas as pd
import os
import numpy as np
import torch
from argparse import ArgumentParser
from internals.drive_data import Drive_iterator
from internals.evaluate import setup, run, print_global_error_all_files
from internals.common import compute_residual_matrix
from model import Gnss_single_epoch_net, get_test_drives


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
        "-m", "--model_path", 
        type=str, 
        default="./single_epoch_network.pt",
        help="Path to the trained model. (default: %(default)s)"
        )
    parser.add_argument(
        "--kf",
        default=False,
        type=bool,
        action="store_true",
        help="Whether to use a Kalman filter for trajectory estimation instead of just weighted least squares. (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="./single_epoch_evaluation",
        help="Directory to save evaluation outputs. (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--drive",
        type=str,
        help="Specific drive to evaluate - If not provided, will evaluate on all test drives. Must be a directory within the base path. (e.g. '2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl')",
        default=None
    )
    args = parser.parse_args()
    base_path = args.base_path
    
    test_drives = get_test_drives(base_path)
    if args.drive:
        drive_paths = [os.path.join(base_path, args.drive)]
    
    drive_iter = Drive_iterator(
        drive_paths=test_drives,
        mode="test"
    )

    save_dir = args.output_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup(Gnss_single_epoch_net, args.model_path, kf=args.kf)

    results = []
    for drive in drive_iter:
        print(f"Evaluating file: {drive.get_directory_name()}")
        error_dict = run(drive, get_feats, save_dir, "single_epoch")
        results.append(error_dict)

    print_global_error_all_files(results)

#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-02-24-18-29-us-ca-lax-o')
#     data_iter = Data_file_iterator(base_path, split='train', prefix='2021-07-14-20-50-us-ca-mtv-e')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2022-04-01-18-22-us-ca-lax-t')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-09-06-00-01-us-ca-routen')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-03-08-21-34-us-ca-mtv-u')
#     # data_iter = Data_file_iterator(base_path, split='train', prefix='2023-05-09-21-32-us-ca-mtv-pe1')
    # drive = Drive(os.path.join(base_path, "2022-02-24-18-29-us-ca-lax-o", "mi8")) # massive los and multipath
    # drive = Drive(os.path.join(base_path, "2020-06-25-00-34-us-ca-mtv-sb-101", "pixel4xl"))
    # drive = Drive(os.path.join(base_path, "2023-09-06-00-01-us-ca-routen", "pixel4xl"))
    # drive = Drive(os.path.join(base_path, "2023-03-08-21-34-us-ca-mtv-u", "pixel7pro")) # test candidate (makes it look good)
    # drive = Drive(os.path.join(base_path, "2021-12-08-20-28-us-ca-lax-c", "sm-g988b")) # test candidate (makes it look good)
    # drive = Drive(os.path.join(base_path, "2021-12-08-18-52-us-ca-lax-b", "pixel6pro"))
    # drive = Drive(os.path.join(base_path, "2023-05-24-20-26-us-ca-sjc-ge2", "pixel7pro")) # train candidate
    # drive = Drive(os.path.join(base_path, "2023-05-19-20-10-us-ca-mtv-ie2", "sm-s908b")) # weirdly bad