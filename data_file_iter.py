from __future__ import annotations
import pandas as pd
import os
import random
from typing import Optional, List

# Small utility to iterate through all device_gnss.csv files in the dataset structure
# [train/test]/[drive_id]/[phone_name]/device_gnss.csv



class Data_file_iterator:
    def __init__(self, base_path: str, split: str='train', limit: int | None=None, 
                 prefix: Optional[str]=None, seed: int = 42, preprocessed: bool = True):
        self.base_path = base_path
        self.split = split
        self.limit = limit
        self.prefix = prefix
        self.abs_counter = 0
        self.current_file = ""

        # Gather all potential files
        self.all_files: List[str] = []
        split_path = os.path.join(base_path, split)
        for drive_id in os.listdir(split_path):
            if self.prefix and self.prefix not in drive_id:
                continue
            drive_path = os.path.join(split_path, drive_id)
            for phone_name in os.listdir(drive_path):
                if preprocessed:
                    file_path = os.path.join(drive_path, phone_name, "device_gnss_preprocessed.csv")
                    # if not os.path.exists(file_path):  # cant do this here due to the limitations set when reading the file in __next__()
                        # assert False, f"Preprocessed file {file_path} does not exist. Set preprocessed=False or run preprocess.py first."
                else:
                    file_path = os.path.join(drive_path, phone_name, "device_gnss.csv")
                truth_file = os.path.join(drive_path, phone_name, "ground_truth.csv")
                if not os.path.exists(file_path) or not os.path.exists(truth_file):
                    continue
                self.all_files.append(file_path)

        # Shuffle with a seed
        random.seed(seed)
        random.shuffle(self.all_files)

        # Index to keep track
        self.file_index = 0

    def __iter__(self) -> Data_file_iterator:
        return self

    def __next__(self) -> pd.DataFrame:
        while self.file_index < len(self.all_files):
            if self.limit is not None and self.abs_counter >= self.limit:
                raise StopIteration
            self.current_file = self.all_files[self.file_index]
            self.file_index += 1

            # Load truth file
            truth = self.get_truth_file()
            if truth["AltitudeMeters"].isna().any():
                continue # skip files with missing ground truth altitude

            df = pd.read_csv(self.current_file, low_memory=False)
            if not (df['MultipathIndicator'] == 1).any():
                continue # skip files with no multipath at all

            self.abs_counter += 1
            return df

        raise StopIteration

    def get_current_file(self) -> str:
        return self.current_file

    def get_truth_file(self) -> pd.DataFrame:
        truth_file = os.path.join(os.path.dirname(self.current_file), "ground_truth.csv")
        return pd.read_csv(truth_file)
    
    def get_stats_file(self) -> pd.DataFrame:
        stats_file = os.path.join(self.base_path, "dataset_stats.csv")
        if not os.path.exists(stats_file):
            assert False, "Dataset stats file does not exist. Run preprocess.py or dataset_stats.py first."
        return pd.read_csv(stats_file)