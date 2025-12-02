import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class Standardiser(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    
class Residual_encoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.res_enc = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (N, N) pseudorange-residual matrix for one epoch.

        Returns:
            row_embeddings: (N, hidden_dim) fixed-size embedding per row.
        """
        # Each row becomes a sequence of N "timesteps" (satellites) with 1 feature
        # Shape: (N_rows, seq_len, feat_dim) = (N, N, 1)
        sequences = M.unsqueeze(-1)

        # Run all rows through the shared LSTM
        lstm_out, _ = self.res_enc(sequences) # (N, N, hidden_dim)

        # Take final hidden state as fixed-size embedding
        row_embeddings = lstm_out[:, -1, :] # (N, hidden_dim)

        return row_embeddings

class Epoch_processor(nn.Module):
    def __init__(self, hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim, # combined feature + residual embeddings
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, feat_dim) satellite features for one epoch.

        Returns:
            lstm_out: (N, hidden_dim) processed satellite features.
        """
        # Add sequence dimension
        x_seq = x.unsqueeze(0) # (1, N, feat_dim)

        lstm_out, _ = self.lstm(x_seq) # (1, N, hidden_dim)

        return lstm_out.squeeze(0) # (N, hidden_dim)

class Network(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, input_dim: int = 13, hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.res_enc = Residual_encoder(
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout
        )

        assert mean.shape[0] == input_dim, "Mean vector size must match input dimension"
        assert std.shape[0] == input_dim, "Std vector size must match input dimension"

        self.input_fc = nn.Sequential(
            Standardiser(
                mean=mean,
                std=std
            ),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Epoch_processor(
                hidden_dim=hidden_dim,
                lstm_layers=lstm_layers,
                dropout=dropout
            )
        )

        # Weight prediction branch
        self.weight_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Pseudorange correction branch
        self.prc_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.prc_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch_sat_data: list[torch.Tensor], residual_matrix: list[torch.Tensor]) -> tuple[list[torch.Tensor],list[torch.Tensor]]:
        """
        Args:
            batch_sat_data: list of tensors [(N_i, F)] satellite features per epoch.
            residual_matrix: list of tensors [(N_i, N_i)] pseudorange residual matrices per epoch.

        Returns:
            weights: list of tensors [(N_i, 1)] predicted weights per satellite per epoch.
            pr_corrections: list of tensors [(N_i, 1)] predicted pseudorange corrections per satellite per epoch.
        """
        assert len(batch_sat_data) == len(residual_matrix), "Batch size of satellite data and residual matrices must match"
        
        weights = []
        pr_corrections = []

        for sat_data, res_matrix in zip(batch_sat_data, residual_matrix):
            # Encode pseudorange residuals
            res_embeddings = self.res_enc(res_matrix) # (N, hidden_dim)

            # Process satellite features
            feat_embeddings = self.input_fc(sat_data) # (N, hidden_dim)

            # Combine embeddings
            combined = torch.cat([feat_embeddings, res_embeddings], dim=-1) # (N, hidden_dim + hidden_dim)

            # Predict weights
            w = self.weight_fc(combined) # (N, 1)
            weights.append(torch.diag(w)) # (N, N) diagonal weight matrix

            # Predict pseudorange corrections
            prc = self.prc_fc(combined) * self.prc_scale # (N, 1)
            pr_corrections.append(prc)

        return weights, pr_corrections