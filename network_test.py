import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class Satellite_weight_network(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Simple feed-forward embedding
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        bidirectional = False
        bi_multi = 2 if bidirectional else 1

        # BiLSTM to model relationships between satellites within each epoch
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention projection
        self.attn_fc = nn.Linear(bi_multi * hidden_dim, 1, bias=False)

        # Output layer: predict weight (bounded 0–1)
        self.output_fc = nn.Sequential(
            nn.Linear((2 * bi_multi) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_sat_data: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch_sat_data: list of tensors [(N_i, F)], where F=8, one per epoch.

        Returns:
            weights_diag: list of (N_i, N_i) diagonal matrices of predicted weights.
        """

        device = next(self.parameters()).device
        lengths = [x.shape[0] for x in batch_sat_data]
        n_max = max(lengths)

        # Pad and stack to (B, N_max, F)
        padded = pad_sequence(batch_sat_data, batch_first=True)  # automatic padding with 0.0
        masks = torch.zeros((len(batch_sat_data), n_max), dtype=torch.bool, device=device)
        for i, n in enumerate(lengths):
            masks[i, :n] = True

        padded = padded.to(device)

        # Encode features
        x = self.input_fc(padded)  # (B, N_max, hidden_dim)

        # Pack sequence to ignore padding during LSTM
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, N_max, 2*hidden_dim)

        # Attention pooling per epoch
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (B, N_max)
        attn_scores = attn_scores.masked_fill(~masks, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_pooled = torch.sum(attn_weights * lstm_out, dim=1, keepdim=True)  # (B, 1, 2*hidden_dim)

        # Combine contextual + pooled info for prediction
        combined = torch.cat([lstm_out, attn_pooled.expand_as(lstm_out)], dim=-1)
        combined = self.output_fc(combined).squeeze(-1)  # (B, N_max)

        # Mask invalid (padded) positions and bound to (0,1)
        combined = combined.masked_fill(~masks, 0.0)
        weights = 1 / (1 + torch.abs(combined))  # smooth bounded weight

        # Build diagonal weight matrices
        weights_diag = [
            torch.diag_embed(weights[i, :n]) for i, n in enumerate(lengths)
        ]

        return weights_diag
