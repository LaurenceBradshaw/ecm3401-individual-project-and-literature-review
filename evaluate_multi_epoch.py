import pandas as pd
import os
import numpy as np
import torch
from argparse import ArgumentParser
from internals.drive_data import Drive_iterator
from internals.epoch_manager import Epoch_manager
from internals.evaluate import setup, run, print_global_error_all_files
from internals.common import compute_residual_matrix
from model import Gnss_multi_epoch_net, get_test_drives

epoch_manager = None

def get_feats(epoch_df: pd.DataFrame, features: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the features for the current epoch, including the temporal history of features for each satellite.

    Parameters
    ----------
    epoch_df: pd.DataFrame
        The dataframe containing the device GNSS data for the current epoch.
    features: list[str]
        The list of feature column names to be extracted.
    device: str
        The device on which to place the tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the padded tensor of features and the lengths tensor.
    """
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
        "-m", "--model_path", 
        type=str, 
        default="./multi_epoch_network.pt",
        help="Path to the trained model. (default: %(default)s)"
        )
    parser.add_argument(
        "--kf",
        default=False,
        action="store_true",
        help="Whether to use a Kalman filter for trajectory estimation instead of just weighted least squares. (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="./multi_epoch_evaluation",
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
        test_drives = [os.path.join(base_path, args.drive)]
    
    drive_iter = Drive_iterator(
        drive_paths=test_drives,
        mode="test"
    )

    save_dir = args.output_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup(Gnss_multi_epoch_net, args.model_path, kf=args.kf)

    kf_string = "kf" if args.kf else "no_kf"
    save_postfix = "multi_epoch"

    if os.path.exists(os.path.join(save_dir, f"results_{kf_string}_{save_postfix}.txt")):
        os.remove(os.path.join(save_dir, f"results_{kf_string}_{save_postfix}.txt"))

    results = []
    for drive in drive_iter:
        print(f"Evaluating file: {drive.get_directory_name()}")
        epoch_manager = Epoch_manager(10)
        error_dict = run(drive, get_feats, save_dir, save_postfix)
        results.append(error_dict)
    
    print_global_error_all_files(results, save_dir, kf_string, save_postfix)
