from __future__ import annotations
import pandas as pd
import os

# Small utility to iterate through all device_gnss.csv files in the dataset structure
# [train/test]/[drive_id]/[phone_name]/device_gnss.csv

class Data_file_iter:
    def __init__(self, base_path: str, split: str='train'):
        self.base_path = base_path
        self.split = split
        self.drive_ids = os.listdir(f"{self.base_path}/{self.split}")
        self.drive_index = 0
        self.phone_names = []
        self.phone_index = 0
        self.current_file = ""

    def __iter__(self) -> Data_file_iter:
        return self

    def __next__(self) -> pd.DataFrame:
        while self.drive_index < len(self.drive_ids):
            drive_id = self.drive_ids[self.drive_index]
            drive_path = f"{self.base_path}/{self.split}/{drive_id}"

            if not self.phone_names:
                self.phone_names = os.listdir(drive_path)
                self.phone_index = 0

            while self.phone_index < len(self.phone_names):
                phone_name = self.phone_names[self.phone_index]
                self.current_file = f"{drive_path}/{phone_name}/device_gnss.csv"
                self.phone_index += 1

                try:
                    df = pd.read_csv(self.current_file)
                    return df
                except FileNotFoundError:
                    continue
                except pd.errors.EmptyDataError:
                    continue

            self.drive_index += 1
            self.phone_names = []

        raise StopIteration
    
    def get_current_file(self) -> str:
        return self.current_file
