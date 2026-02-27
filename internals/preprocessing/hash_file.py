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

def preprocessing_artifacts_path():
    if not os.path.exists("preprocessing_artifacts"):
        os.makedirs("preprocessing_artifacts")
        
    return "preprocessing_artifacts"