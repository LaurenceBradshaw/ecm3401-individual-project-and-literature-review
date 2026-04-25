import numpy as np
from pathlib import Path
from internals import common
from internals.drive_data import Drive, Drive_iterator
from internals.constants import PR_COL, SAT_POS_COLS
import internals.gnss_positioning as gp

def compute_drive_max_residual(drive: Drive) -> float:
    """
    Given a drive, compute the maximum norm of the pseudorange residuals across all epochs.

    Parameters
    ---------
    drive: Drive

    Returns
    -------
    float
        The maximum norm of the pseudorange residuals for the drive.
    """
    df, truth_df = drive.get_dataframes()

    residual_norms = []

    for epoch_id, epoch_df in df.groupby("epoch_id"):
        pos_truth, vel_truth = common.get_ground_truth(truth_df, epoch_df)

        pr = epoch_df[PR_COL].to_numpy()
        sat_pos = epoch_df[SAT_POS_COLS].to_numpy()

        residual = gp.pr_residuals(pos_truth.numpy().flatten(), sat_pos, pr)
        residual_norms.append(np.linalg.norm(residual))

    if len(residual_norms) == 0:
        return 0.0

    return float(np.max(residual_norms))

def remove_base_path(paths: list[str], base_path: str) -> list[str]:
    """
    Remove the base path from a list of paths to get relative paths.

    Parameters
    ----------
    paths: list[str]
        A list of file paths to be processed.
    base_path: str
        The base directory to be removed from the paths.

    Returns
    -------
    list[str]
        A list of relative paths.
    """
    base = Path(base_path)

    return [
        str(Path(p).relative_to(base))
        for p in paths
    ]

def find_test_set(drive_iter: Drive_iterator, base_path: str) -> list[str]:
    """
    Find a test set of drives based on their maximum pseudorange residuals.
    Compute the maximum residual for each drive, sort them, and select a subset of drives using a doubling range.

    Parameters
    ----------
    drive_iter: Drive_iterator
        An iterator over the drives in the dataset.
    base_path: str
        The base directory containing drive data subdirectories.

    Returns
    -------
    list[str]
        A list of relative paths for the selected test drives.
    """
    print("Computing max residuals for each drive to select test set...")
    drive_residuals = []
    for drive in drive_iter:
        max_residual = compute_drive_max_residual(drive)
        drive_residuals.append((drive.get_directory_name(), max_residual))
    
    # Sort drives by max residual in descending order
    drive_residuals.sort(key=lambda x: x[1], reverse=True)

    # Take doubling range of drives as test set
    print("Selecting test drives based on max residuals using a doubling range...")
    test_drives = []
    i = 1
    while i < len(drive_residuals):
        test_drives.append(drive_residuals[i][0])
        i = i * 2

    return remove_base_path(test_drives, base_path)