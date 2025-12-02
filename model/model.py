import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from standardise_layer import Standardiser


class Gnss_net(nn.Module):
    """
    Inputs:
      - sat_feats_padded: tensor (batch, N_max, 5)
      - lengths: 1D tensor (batch,) of ints with true number of satellites N_i (<= N_max)
    Behaviour:
      - Per-item branch: 3 x [Linear(5->64) / ReLU]
      - LSTM branch: single-layer, unidirectional LSTM (hidden_size=360 by default)
      - LSTM final hidden state is concatenated to each per-item embedding
      - Two heads (visibility & correction): each 2-layer MLP (64 hidden), visibility ends in Sigmoid
    Outputs:
      - visibility: (batch, N_max) with values in [0,1]; padded positions set to 0 but use mask to ignore them.
      - correction: (batch, N_max) raw scalar (no activation); padded positions set to 0 but use mask to ignore them.
      - mask: boolean tensor (batch, N_max) where True means valid (not padding)
    """

    def __init__(
        self,
        mean: torch.Tensor, 
        std: torch.Tensor,
        input_dim: int = 5,
        per_item_hidden: int = 64,
        per_item_layers: int = 3,
        lstm_hidden_size: int = 360,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.per_item_hidden = per_item_hidden
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional

        # --- Per-item branch: sequential 3 layers of size 64 with ReLU ---
        per_item_layers_seq = [Standardiser(mean, std)]
        in_dim = input_dim
        for _ in range(per_item_layers):
            per_item_layers_seq.append(nn.Linear(in_dim, per_item_hidden))
            per_item_layers_seq.append(nn.ReLU())
            in_dim = per_item_hidden
        # final per-item embedding size == per_item_hidden
        self.per_item_net = nn.Sequential(*per_item_layers_seq)

        # --- LSTM branch ---
        # single-layer, unidirectional by default to match the paper
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )

        lstm_out_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.lstm_out_dim = lstm_out_dim

        # --- Output heads ---
        # Combined feature size: per_item_hidden + lstm_out_dim
        combined_dim = per_item_hidden + lstm_out_dim

        # Visibility head: [Linear(combined->64), ReLU, Linear(64->1), Sigmoid]
        self.visibility_head = nn.Sequential(
            nn.Linear(combined_dim, per_item_hidden),
            nn.ReLU(),
            nn.Linear(per_item_hidden, 1),
            nn.Sigmoid(),
        )

        # Correction head: [Linear(combined->64), ReLU, Linear(64->1)] (no activation)
        self.correction_head = nn.Sequential(
            nn.Linear(combined_dim, per_item_hidden),
            nn.ReLU(),
            nn.Linear(per_item_hidden, 1),
        )

    def forward(self, sat_feats_padded: torch.Tensor, lengths: torch.Tensor):
        """
        sat_feats_padded: (batch, N_max, 5) - float tensor
        lengths: (batch,) - int tensor with actual counts N_i (<= N_max)
        Returns:
          visibility (batch, N_max), correction (batch, N_max), mask (batch, N_max)
        """
        batch, n_max, feat_dim = sat_feats_padded.shape
        device = sat_feats_padded.device
        assert feat_dim == self.input_dim, f"expected input_dim={self.input_dim}, got {feat_dim}"

        # mask of valid positions
        arange = torch.arange(n_max, device=device).unsqueeze(0)  # (1, N_max)
        mask = arange < lengths.unsqueeze(1)  # (batch, N_max), bool

        # --- Per-item branch ---
        # flatten padded items -> process with per_item_net -> reshape back
        flat = sat_feats_padded.view(batch * n_max, feat_dim)  # (batch*N_max, 5)
        per_item_out_flat = self.per_item_net(flat)  # (batch*N_max, per_item_hidden)
        per_item_out = per_item_out_flat.view(batch, n_max, self.per_item_hidden)  # (batch, N_max, 64)

        # --- LSTM branch (use pack to ignore padding) ---
        # Need to sort by lengths descending for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        sorted_lengths, sort_idx = torch.sort(lengths_cpu, descending=True)
        unsort_idx = torch.argsort(sort_idx)

        sat_sorted = sat_feats_padded[sort_idx.to(device)]  # (batch, N_max, feat_dim)
        # pack. We must provide actual lengths (on CPU) for pack_padded_sequence
        packed = pack_padded_sequence(sat_sorted, sorted_lengths.tolist(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed)

        # h_n shape: (num_layers * num_directions, batch_sorted, hidden_size)
        # take last layer's hidden state
        num_directions = 2 if self.lstm_bidirectional else 1
        # get last layer index
        last_layer_idx = self.lstm_num_layers - 1
        # compute index in h_n
        # reshape to (num_layers, num_directions, batch, hidden_size)
        h_n_view = h_n.view(self.lstm_num_layers, num_directions, -1, self.lstm_hidden_size)
        last_layer_h = h_n_view[last_layer_idx]  # (num_directions, batch_sorted, hidden_size)

        if self.lstm_bidirectional:
            # concat forward & backward => (batch_sorted, hidden*2)
            lstm_feat_sorted = torch.cat([last_layer_h[0], last_layer_h[1]], dim=1)
        else:
            lstm_feat_sorted = last_layer_h.squeeze(0)  # (batch_sorted, hidden_size)

        # unsort to original batch order
        lstm_feat = lstm_feat_sorted[unsort_idx.to(device)]  # (batch, lstm_out_dim)

        # expand lstm_feat to match per-item count: (batch, 1, lstm_out_dim) -> (batch, N_max, lstm_out_dim)
        lstm_expanded = lstm_feat.unsqueeze(1).expand(-1, n_max, -1)

        # --- Concatenate per-item + lstm for each satellite ---
        combined = torch.cat([per_item_out, lstm_expanded], dim=2)  # (batch, N_max, combined_dim)

        # --- Predict heads per item ---
        # Flatten to feed heads (they produce scalar outputs)
        combined_flat = combined.view(batch * n_max, -1)  # (batch*N_max, combined_dim)

        vis_flat = self.visibility_head(combined_flat)  # (batch*N_max, 1) in (0,1)
        corr_flat = self.correction_head(combined_flat)  # (batch*N_max, 1) raw

        # reshape back (batch, N_max)
        visibility = vis_flat.view(batch, n_max)
        correction = corr_flat.view(batch, n_max)

        # Zero out padded positions for safety (use mask to ignore during loss calculation)
        visibility = visibility * mask.float()
        correction = correction * mask.float()

        return visibility, correction, mask
