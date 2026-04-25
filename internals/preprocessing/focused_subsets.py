import numpy as np
import os
import internals.common as common
import internals.gnss_positioning as gp
from internals.drive_data import Drive
from internals.constants import PR_COL, SAT_POS_COLS, PRR_COL, SAT_VEL_COLS
from internals.preprocessing.hash_file import hash_file

HASH_FILE = "preprocess_focused_subsets_hash_table.json"

def create_focused_subset(drive: Drive, drive_i: int, n_drives: int, hash_table: dict, per_file_hash: str) -> None:
    """
    Split a drive into 3 parts based on the largest pseudorange residuals, and save these parts as new files.

    Parameters
    ----------
    drive: Drive
        The drive to be processed.
    drive_i: int
        The index of the current drive (for progress tracking).
    n_drives: int
        The total number of drives (for progress tracking).
    hash_table: dict
        A dictionary to store hashes for caching purposes.
    per_file_hash: str
        A hash representing the current state of the preprocessing logic.

    Returns
    -------
    None
    """
    hash = hash_file(__file__)
    joint_hash = hash + per_file_hash # join the hashes that way if per_file changes, but this focused_subset doesn't, we know we still need to redo this drive

    window_size = 100
    half_window = window_size // 2
    n_parts = 3

    if f"parts_{drive.get_directory_name()}" in hash_table and hash_table[f"parts_{drive.get_directory_name()}"] == joint_hash:
        print(f"[{drive_i+1}/{n_drives}] Skipping drive for focused subsets (already processed with same code): {drive.get_directory_name()}")
        return
    elif f"parts_{drive.get_directory_name()}" in hash_table and hash_table[f"parts_{drive.get_directory_name()}"] != joint_hash:
        # Delete all existing part files for this drive.
        for part_idx in range(1, n_parts + 1):
            part_path = os.path.join(drive.get_directory_name(), f"device_gnss_preprocessed_part{part_idx}.csv")
            if os.path.exists(part_path):
                os.remove(part_path)

    print(f"[{drive_i+1}/{n_drives}] Processing drive for focused subsets: {drive.get_directory_name()}")
    drive_i += 1
    df, truth_df = drive.get_dataframes()
    
    residuals = []
    epoch_ids = []

    # Compute residuals per epoch
    for i, epoch_df in df.groupby('epoch_id'):
        pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)

        pr = epoch_df[PR_COL].to_numpy()
        sat_pos = epoch_df[SAT_POS_COLS].to_numpy()

        residual = gp.pr_residuals(pos_truth.numpy().flatten(), sat_pos, pr)
        residuals.append(residual)
        epoch_ids.append(i)

    residuals_array = np.array([np.linalg.norm(r) for r in residuals])
    used_mask = np.zeros_like(residuals_array, dtype=bool) # Track epochs already in windows

    for part_idx in range(1, n_parts + 1):
        # Mask already used epochs
        masked_residuals = np.where(used_mask, -1, residuals_array)
        if masked_residuals.size <= 0:
            break
        if masked_residuals.max() < 0:
            break # No remaining residuals

        idx_max = masked_residuals.argmax()

        # Determine window boundaries
        start = max(0, idx_max - half_window)
        end = min(len(residuals), idx_max + half_window)

        # Mark these epochs as used
        used_mask[start:end] = True

        selected_epochs = epoch_ids[start:end]

        df_to_save = df[df['epoch_id'].isin(selected_epochs)].copy()

        out_path = os.path.join(drive.get_directory_name(), f"device_gnss_preprocessed_part{part_idx}.csv")
        df_to_save.to_csv(out_path, index=False)

    # Update hash table
    hash_table[f"parts_{drive.get_directory_name()}"] = joint_hash