import hashlib
import os
from pathlib import Path

def hash_file(file: str) -> str:
    """
    Computes the SHA-256 hash of a file's contents.

    Parameters
    ----------
    file: str
        The path to the file to be hashed.

    Returns
    -------
    str
        The hexadecimal representation of the file's SHA-256 hash.
    """
    file_path = Path(file).resolve()
    hasher = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def preprocessing_artifacts_path():
    """
    The path where preprocessing artifacts (like hash tables) will be stored. 
    Creates the directory if it doesn't exist.

    Returns
    -------
    str
        The path to the preprocessing artifacts directory.
    """
    if not os.path.exists("preprocessing_artifacts"):
        os.makedirs("preprocessing_artifacts")
        
    return "preprocessing_artifacts"