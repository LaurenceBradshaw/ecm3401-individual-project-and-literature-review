import hashlib
import os
from pathlib import Path

def hash_file(file: str) -> str:
    file_path = Path(file).resolve()
    hasher = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def hash_store_path():
    if not os.path.exists("preprocessing_hashes"):
        os.makedirs("preprocessing_hashes")
        
    return "preprocessing_hashes"