import torch
import torch.nn as nn

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