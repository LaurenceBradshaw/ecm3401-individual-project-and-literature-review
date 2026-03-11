import json
import os
from internals.drive_data import Drive_iterator
from internals.preprocessing.hash_file import preprocessing_artifacts_path
from internals.preprocessing.per_file import process_file, per_file_hash
from internals.preprocessing.per_file import HASH_FILE as PER_FILE_HASH_FILE
from internals.preprocessing.dataset_stats import create_dataset_stats
from internals.preprocessing.dataset_stats import HASH_FILE as DATASET_STATS_HASH_FILE
from internals.preprocessing.focused_subsets import create_focused_subset
from internals.preprocessing.focused_subsets import HASH_FILE as FOCUSED_SUBSETS_HASH_FILE
from internals.preprocessing.find_test_set import find_test_set

# This file is wildly inefficient since it loops through the data multiple times.

def load_hash_table(file_name: str) -> dict:
    hash_dir = preprocessing_artifacts_path()
    if not os.path.exists(os.path.join(hash_dir, file_name)):
        return {}
    else:
        with open(os.path.join(hash_dir, file_name), "r") as f:
            return json.load(f)
        
def write_hash_table(file_name: str, hash_table: dict) -> None:
    hash_dir = preprocessing_artifacts_path()
    with open(os.path.join(hash_dir, file_name), "w") as f:
        json.dump(hash_table, f)

def save_string_list(file_path: str, string_list: list[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for item in string_list:
            f.write(item)
            f.write("\n")

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023/train"
    
    drive_iter = Drive_iterator(
        drive_paths=[
            os.path.join(base_path, d, p)
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
            for p in os.listdir(os.path.join(base_path, d))
            if os.path.isdir(os.path.join(base_path, d, p))
        ],
        preprocessed=False
    )

    per_file_hash_table = load_hash_table(PER_FILE_HASH_FILE)

    n_drives = drive_iter.nitems()
    drive_i = 0
    for drive in drive_iter:
        process_file(drive, drive_i, n_drives, per_file_hash_table)
        drive_i += 1
        write_hash_table(PER_FILE_HASH_FILE, per_file_hash_table)
    
    current_per_file_hash = per_file_hash()

    print("Pre-processing complete")
    dataset_stats_hash_table = load_hash_table(DATASET_STATS_HASH_FILE)
    print("Creating dataset statistics...")
    create_dataset_stats(base_path, os.path.join(base_path, "dataset_stats.csv"), dataset_stats_hash_table, current_per_file_hash)
    write_hash_table(DATASET_STATS_HASH_FILE, dataset_stats_hash_table)

    focused_subset_hash_table = load_hash_table(FOCUSED_SUBSETS_HASH_FILE)

    drive_iter = Drive_iterator(
        drive_paths=[
            os.path.join(base_path, d, p)
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
            for p in os.listdir(os.path.join(base_path, d))
            if os.path.isdir(os.path.join(base_path, d, p))
        ],
        preprocessed=True
    )

    drive_i = 0
    n_drives = drive_iter.nitems()
    for drive in drive_iter:
        create_focused_subset(drive, drive_i, n_drives, focused_subset_hash_table, current_per_file_hash)
        drive_i += 1
        write_hash_table(FOCUSED_SUBSETS_HASH_FILE, focused_subset_hash_table)

    drive_iter = Drive_iterator(
        drive_paths=[
            os.path.join(base_path, d, p)
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
            for p in os.listdir(os.path.join(base_path, d))
            if os.path.isdir(os.path.join(base_path, d, p))
        ],
        preprocessed=True
    )

    test_drives = find_test_set(drive_iter, base_path)
    print(f"Selected test drives: {test_drives}")
    hash_dir = preprocessing_artifacts_path() # Put in here and treat as preprocessing artefact.
    save_string_list(os.path.join(hash_dir, "test_drives.txt"), test_drives)

    

        



    
    
