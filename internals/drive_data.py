import os
from typing import Iterator, List, Tuple
import pandas as pd


class Drive:
    def __init__(self, drive_path: str, preprocessed: bool) -> None:
        self.drive_path_ = drive_path
        self.preprocessed_ = preprocessed

        if not os.path.exists(self.drive_path_):
            raise FileNotFoundError(f"Path does not exist: {self.drive_path_}")

        if not os.path.isdir(self.drive_path_):
            raise NotADirectoryError(f"Path is not a directory: {self.drive_path_}")

        self.device_gnss_df_ = None
        self.ground_truth_df_ = None

        self._load_files()

    def _load_files(self) -> None:
        if self.preprocessed_:
            device_gnss_filename = "device_gnss_preprocessed.csv"
        else:
            device_gnss_filename = "device_gnss.csv"

        device_gnss_path = os.path.join(self.drive_path_, device_gnss_filename)
        ground_truth_path = os.path.join(self.drive_path_, "ground_truth.csv")

        if not os.path.isfile(device_gnss_path):
            raise FileNotFoundError(f"Missing device gnss file: {device_gnss_path}")

        if not os.path.isfile(ground_truth_path):
            raise FileNotFoundError(f"Missing ground truth file: {ground_truth_path}")

        self.device_gnss_df_ = pd.read_csv(device_gnss_path, low_memory=False)
        self.ground_truth_df_ = pd.read_csv(ground_truth_path, low_memory=False)

        if self.ground_truth_df_["AltitudeMeters"].isna().any():
            raise ValueError(f"Ground truth file contains missing altitude values: {ground_truth_path}")
        
        if not (self.device_gnss_df_['MultipathIndicator'] == 1).any():
            print(f"Warning: No MultipathIndicator == 1 in device gnss file: {device_gnss_path}")

    def get_directory_name(self) -> str:
        return os.path.normpath(self.drive_path_)

    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.device_gnss_df_, self.ground_truth_df_


class Drive_iterator:
    def __init__(self, drive_paths: List[str], preprocessed: bool = True) -> None:
        self.drive_paths_ = drive_paths
        self.preprocessed_ = preprocessed
        self.index_ = 0

    def __iter__(self) -> Iterator[Drive]:
        self.index_ = 0
        return self

    def __next__(self) -> Drive:
        if self.index_ >= len(self.drive_paths_):
            raise StopIteration

        drive_path = self.drive_paths_[self.index_]
        self.index_ += 1

        try:
            drive = Drive(drive_path, self.preprocessed_)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"Skipping drive due to error: {e}")
            return self.__next__()

        return drive
    
    def nitems(self) -> int:
        return len(self.drive_paths_)
    
