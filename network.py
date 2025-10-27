import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch import nn

class Pseudorange_residual_encoder(nn.Module):
    def __init__(self, lstm_hidden: int = 32, lstm_layers: int = 1, attn_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        # Bidirectional LSTM maps (N, 1) -> (N, 2 * hidden)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Additive attention: score_t = v^T * tanh(W * h_t)
        self.attn_w = nn.Linear(2 * lstm_hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        self.norm = nn.LayerNorm(2 * lstm_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, M: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        """
        Args:
            M: (N, N) pseudorange-residual matrix for one epoch.

        Returns:
            row_embeddings: (N, 2 * lstm_hidden) fixed-size embedding per row.
        """
        # Each row becomes a sequence of N "timesteps" (satellites) with 1 feature
        # Shape: (N_rows, seq_len, feat_dim) = (N, N, 1)
        sequences = M.unsqueeze(-1)

        # Run all rows through the shared Bi-LSTM
        lstm_out, _ = self.lstm(sequences) # (N, N, 2 * hidden)

        # Attention scores for each "timestep" (satellite) in the sequence
        attn_proj = torch.tanh(self.attn_w(lstm_out)) # (N, N, attn_dim)
        attn_scores = self.attn_v(attn_proj).squeeze(-1) # (N, N)

        # Apply mask before softmax
        if mask is not None:
            # mask: (N,) -> (1, N) for broadcasting
            seq_len = attn_scores.size(1)
            # Trim or pad mask accordingly
            mask = mask[:seq_len]
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

        # Normalised attention weights over the time dimension
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1) # (N, N, 1)

        # Weighted sum of LSTM outputs (attention pooling)
        pooled = torch.sum(attn_weights * lstm_out, dim=1) # (N, 2 * hidden)

        # Normalise and regularise
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        return pooled
    
class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()
        self.importance_encoder = Pseudorange_residual_encoder(lstm_layers=2)
        self.lstm = nn.LSTM(input_size, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(512, 1)  # Output a single value per satellite

        # [importance_embedding(16), seen_t_minus_1, age_since_last_obs, DOP(5), sin(el), cos(el), sin(az), cos(az)
        # lock_time, mean Cn0DbHz_linear, var Cn0DbHz_linear, Cn0DbHz_linear, window_size, PDD_scores, contellation_one_hot(7)]
    def forward(self, residuals: list[np.ndarray], rest_of_data: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            residuals: List of (B, N, N) numpy arrays of pseudorange residuals per satellite.
                       Note: N can vary per batch item.
            rest_of_data: (B, N, F) tensor of other features per satellite.

        Returns:
            weights: (B, N, 1) tensor of predicted weight least squares weightings per satellite
        """
        batch_size = len(residuals)
        importance_embeddings = []
        lengths = [len(r) for r in residuals]
        n_max = max(lengths)

        rest_of_data = pad_sequence(rest_of_data, batch_first=True)

        # tensors = [torch.as_tensor(r, dtype=torch.float32, device=rest_of_data.device) for r in residuals]
        # padded = []
        masks = torch.zeros((len(residuals), n_max), dtype=torch.bool, device=rest_of_data.device)
        # for i, t in enumerate(tensors):
        #     n = t.shape[0]
        #     masks[i, :n] = True
        #     if n < n_max:
        #         pad = (0, n_max - n, 0, n_max - n)  # (left, right, top, bottom)
        #         t = F.pad(t, pad, mode='constant', value=0.0)
        #     padded.append(t)
        padded_tensors = []
        for i, r in enumerate(residuals):
            t = torch.tensor(r, dtype=torch.float32, device=rest_of_data.device)
            n = t.shape[0]
            masks[i, :n] = True
            if n < n_max:
                pad = (0, n_max - n, 0, n_max - n)  # pad right and bottom
                t = F.pad(t, pad, mode='constant', value=0.0)
            padded_tensors.append(t)
        residuals_tensor = torch.stack(padded_tensors, dim=0)  # (B, N_max, N_max)

        for i in range(batch_size):
            emb = self.importance_encoder(residuals_tensor[i], masks[i])  # (N_i, embedding_dim)
            importance_embeddings.append(emb)

        # Pad to max length
        padded_embeddings = pad_sequence(importance_embeddings, batch_first=True)  # (B, N_max, E)
        padded_rest = pad_sequence(
            [rest_of_data[i][:lengths[i]] for i in range(batch_size)],
            batch_first=True
        )  # (B, N_max, F)

        # Combine embeddings + other features
        lstm_input = torch.cat([padded_embeddings, padded_rest], dim=-1)  # (B, N_max, input_size)

        # Pack to skip padded timesteps
        packed = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)

        # Run through LSTM
        packed_out, _ = self.lstm(packed)

        # Unpack back to padded sequence
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, N_max, 2 * hidden_size)

        # Predict weights
        weights = self.output_layer(lstm_out)  # (B, N_max, 1)
        weights = weights.masked_fill(~masks.unsqueeze(-1), 0.0)
        weights = 1 / (1 + torch.abs(weights))
        weights = weights.squeeze(-1) 

        # Restore diagonal weight matrices (but only up to real N per batch)
        weights_diag = [
            torch.diag_embed(weights[i, :n]) for i, n in enumerate(lengths)
        ]

        return weights_diag
        # batch_size = len(residuals)
        # importance_embeddings = []

        # for i in range(batch_size):
        #     # res_tensor = torch.tensor(residuals[i], dtype=torch.float32, device=rest_of_data.device)
        #     emb = self.importance_encoder(residuals[i])  # (N, embedding_dim)
        #     importance_embeddings.append(emb)

        # # Pad importance embeddings to create a batch tensor
        # padded_embeddings = nn.utils.rnn.pad_sequence(importance_embeddings, batch_first=True)  # (B, N_max, embedding_dim)

        # # Concatenate importance embeddings with the rest of the data
        # lstm_input = torch.cat([padded_embeddings, rest_of_data], dim=-1)  # (B, N_max, input_size)

        # # Pass through LSTM
        # lstm_out, _ = self.lstm(lstm_input)  # (B, N_max, 2 * hidden_size)

        # # Predict weights
        # weights = self.output_layer(lstm_out)  # (B, N_max, 1)

        # # Apply sigmoid to ensure weights are in (0, 1)
        # weights = 1 / (1 + 1 * torch.abs(weights)) # (B, N_max, 1)

        # # Modify the shape of weights so that each value is on the diagonal of a matrix (B, N, N)
        # weights_vec = weights.squeeze(-1)          # (B, N)
        # weights_diag = torch.diag_embed(weights_vec)  # (B, N, N)

        # return weights_diag