import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class Satellite_weight_network(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        bidirectional = False
        bi_multi = 2 if bidirectional else 1

        self.res_enc = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Simple feed-forward embedding
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # BiLSTM to model relationships between satellites within each epoch
        self.lstm = nn.LSTM(
            input_size=hidden_dim + hidden_dim, # combined feature + residual embeddings
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention projection
        self.attn_fc = nn.Linear(bi_multi * hidden_dim, 1, bias=False)

        # Output layer: predict weight (bounded 0–1)
        # Weight prediction branch
        self.weight_fc = nn.Sequential(
            nn.Linear((2 * bi_multi) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Pseudorange correction branch
        self.prc_fc = nn.Sequential(
            nn.Linear((2 * bi_multi) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.prc_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch_sat_data: list[torch.Tensor], residual_matrix: list[torch.Tensor]) -> tuple[list[torch.Tensor],list[torch.Tensor]]:
        """
        Args:
            batch_sat_data: list of tensors [(N_i, F)]
            residual_matrix: list of tensors [(N_i, N_i-1)] for N_i - 1 pseudorange residuals

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

        res_embeds = []
        for res_mat in residual_matrix:
            n_i = res_mat.shape[0]
            sat_embeds = []

            for j in range(n_i): # encode rows of residual matrix
                seq = res_mat[j].reshape(-1, 1).unsqueeze(0).to(device)  # (1, N_i-1, 1)
                _, (h_n, _) = self.res_enc(seq)  # h_n: (num_layers, 1, hidden_dim)
                sat_embeds.append(h_n[-1].squeeze(0))  # (hidden_dim,)

            res_embeds.append(torch.stack(sat_embeds, dim=0))  # (N_i, hidden_dim)

        # Pad residual embeddings to (B, N_max, hidden_dim)
        res_padded = pad_sequence(res_embeds, batch_first=True).to(device)

        x = torch.cat([x, res_padded], dim=-1)  # combine feature and residual embeddings

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

        weight_logits = self.weight_fc(combined).squeeze(-1)
        weight_logits = weight_logits.masked_fill(~masks, 0.0)
        weights = 1 / (1 + torch.abs(weight_logits))


        weights_diag = [
            torch.diag_embed(weights[i, :n]) for i, n in enumerate(lengths)
        ]

        for i, w in enumerate(weights_diag):
            d = torch.diagonal(w)
            d = d / (d.mean() + 1e-6)
            d_clamped = torch.clamp(d, 1e-3, 1e3)
            weights_diag[i] = torch.diag_embed(d_clamped)

        pr_corrections = self.prc_fc(combined).squeeze(-1)
        pr_corrections = pr_corrections.masked_fill(~masks, 0.0)
        pr_corrections = pr_corrections * self.prc_scale

        pr_corrections_list = [
            pr_corrections[i, :n] for i, n in enumerate(lengths)
        ]

        return weights_diag, pr_corrections_list
