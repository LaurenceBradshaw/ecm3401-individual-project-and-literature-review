from utils.data_file_iter import Data_file_iter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if __name__ == "__main__":
    # For each data file
    # Split into epochs,
    # Apply data transformations like when training
    # Calculate all the required inputs for the current epoch
    # Predict weights for current epoch
    # Run WLS with those weights
    # Plot position
    pass