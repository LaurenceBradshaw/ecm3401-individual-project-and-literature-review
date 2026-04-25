from __future__ import annotations
import os
import pandas as pd


class Drive:
    """
    Represents a single drive, encapsulating the loading and access to its device GNSS data and ground truth data.

    A drive has several settings that determine what data is loaded:
    * Preprocessed: determines whether to load the original "device_gnss.csv" or the preprocessed "device_gnss_preprocessed.csv" (or parts thereof).
    * Mode: determines whether to load the full preprocessed file (for test mode) or the split parts (for train mode).
    """

    def __init__(self, drive_path: str, preprocessed: bool = True, mode: str = "test", part: int = 0) -> None:
        """
        Constructs a Drive object by loading the relevant CSV files from the specified drive directory.

        Parameters
        ----------
        drive_path: str
            The path to the drive directory containing the CSV files.
        preprocessed: bool, optional
            Whether to load preprocessed data, by default True
        mode: str, optional
            The mode of operation ('test' or 'train'), by default "test"
        part: int, optional
            The part number to load in train mode, by default 0

        Raises
        ------
        FileNotFoundError
            If the required CSV files are not found in the drive directory.
        NotADirectoryError
            If the provided drive path is not a directory.
        ValueError
            If the ground truth file contains missing altitude values.
        """
        self.drive_path_ = drive_path
        self.preprocessed_ = preprocessed
        self.mode_ = mode
        self.part_ = part

        if not os.path.exists(self.drive_path_):
            raise FileNotFoundError(f"Path does not exist: {self.drive_path_}")

        if not os.path.isdir(self.drive_path_):
            raise NotADirectoryError(f"Path is not a directory: {self.drive_path_}")

        self.device_gnss_df_ = None
        self.ground_truth_df_ = None
        self._load_files()

    def _load_files(self) -> None:
        """
        Subroutine to load the device GNSS and ground truth CSV files based on the current settings of the Drive object.
        """
        if self.preprocessed_:
            if self.mode_ == "train":
                device_gnss_filename = f"device_gnss_preprocessed_part{self.part_}.csv" # Used in training
            else:
                device_gnss_filename = "device_gnss_preprocessed.csv" # Used in test mode and also preprocessing to produce the part files
        else:
            device_gnss_filename = "device_gnss.csv" # Used initially by preprocessing

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

    def get_directory_name(self) -> str:
        """
        Gets the name of the drive directory.

        Returns
        -------
        str
            The name of the drive directory.
        """
        return os.path.normpath(self.drive_path_)

    def get_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets the loaded device GNSS and ground truth dataframes.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the device GNSS dataframe and the ground truth dataframe.
        """
        return self.device_gnss_df_, self.ground_truth_df_


class Drive_iterator:
    """
    An iterator for a collection of drives, allowing sequential access to each drive's data.
    The iterator it made to support all modes of operation that the Drive class supports.

    * Preprocessed: determines whether to load the original "device_gnss.csv" or the preprocessed "device_gnss_preprocessed.csv" (or parts thereof).
    * Mode: determines whether to load the full preprocessed file (for test mode) or the split parts (for train mode).
    """

    def __init__(self, drive_paths: list[str], preprocessed: bool = True, mode: str = "test") -> None:
        """
        Parameters
        ----------
        drive_paths: list[str]
            A list of paths to the drive directories.
        preprocessed: bool
            Whether to load the preprocessed data.
        mode: str
            The mode of operation (train or test).
        """
        self.drive_paths_ = drive_paths
        self.preprocessed_ = preprocessed
        self.mode_ = mode
        self.index_ = 0
        self.part_num_ = 0
        self.num_parts_ = 0

    def __iter__(self) -> Drive_iterator:
        """
        Returns the iterator object itself.
        """
        self.index_ = 0
        return self

    def __next__(self) -> Drive:
        """
        Returns the next Drive object in the iteration.
        """
        if self.index_ >= len(self.drive_paths_):
            raise StopIteration

        drive_path = self.drive_paths_[self.index_]

        # Discover parts exactly once per drive
        if self.part_num_ == 0 and self.mode_ == "train":
            self.num_parts_ = len([
                f for f in os.listdir(drive_path)
                if f.endswith(".csv") and "_part" in f
            ])

        # Case 1: train mode with remaining parts
        if self.mode_ == "train" and self.part_num_ < self.num_parts_:
            self.part_num_ += 1
            part_num = self.part_num_
        # Case 2: train mode, parts exhausted -> move to next drive
        elif self.mode_ == "train":
            self.part_num_ = 0
            self.num_parts_ = 0
            self.index_ += 1
            return self.__next__()
        # Case 3: not train mode -> load full drive once
        else:
            part_num = 0
            self.index_ += 1

        try:
            return Drive(
                drive_path,
                self.preprocessed_,
                self.mode_,
                part_num,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"Skipping part due to error: {e}")

            # Skip failed part, try next part or drive
            if self.mode_ == "train" and part_num > 0:
                return self.__next__()

            self.part_num_ = 0
            self.num_parts_ = 0
            self.index_ += 1
            return self.__next__()
    
    def nitems(self) -> int:
        """
        Number of drives in the iterator (not accounting for parts, just drive directories).

        Accounting for parts would require a first iteration to count them.
        """
        return len(self.drive_paths_)
    
    def nparts(self) -> int:
        """
        Number of parts in the current drive (only relevant in train mode, otherwise returns 1).
        This could change when swapping to a new drive.

        Returns
        -------
        int
            The number of parts.
        """
        if self.mode_ == "train":
            return self.num_parts_
        else:
            return 1
    
