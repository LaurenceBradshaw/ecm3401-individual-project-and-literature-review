import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.max(torch.abs(x))
    

class Standardiser(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, need_standardising: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("need_standardising", need_standardising)

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        x_std = (x - self.mean) / self.std
        return torch.where(self.need_standardising, x_std, x)
    

class Single_sat_encoder(nn.Module):
    """
    Encodes each satellite independently using a small MLP.
    """
    def __init__(self, feat_dim: int, hidden_dim: int, emb_dim: int, dropout: float):
        super().__init__()
        self.net_ = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_sats, feat_dim)
        return self.net_(x)


class Multi_sat_encoder(nn.Module):
    """
    Encodes the entire satellite set using an LSTM to capture live sky conditions.
    """
    def __init__(self, feat_dim: int, lstm_hidden: int, lstm_layers: int):
        super().__init__()
        self.lstm_ = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=False,
        )

    def forward(self, sats: torch.Tensor) -> torch.Tensor:
        # sats: (num_sats, feat_dim)
        seq = sats.unsqueeze(1)  # (num_sats, 1, feat_dim)
        lstm_out, _ = self.lstm_(seq)
        return lstm_out.squeeze(1)  # (num_sats, lstm_hidden)
    
class Temporal_sat_encoder(nn.Module):
    """
    Encodes the per-satellite time sequence into a fixed-size vector.
    Input shape: (time_steps, feat_dim)
    Output shape: (emb_dim)
    """
    def __init__(self, feat_dim: int, lstm_hidden: int, lstm_layers: int, emb_dim: int, dropout: float):
        super().__init__()

        self.lstm_ = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.proj_ = nn.Sequential(
            nn.Linear(lstm_hidden, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor):
        # seq:     (num_sats, time_steps, feat_dim)
        # lengths: (num_sats,) valid sequence lengths

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm_(packed)

        # final hidden state from last layer
        last_hidden = h_n[-1]  # (num_sats, lstm_hidden)

        return self.proj_(last_hidden)  # (num_sats, emb_dim)
    

class Pairwise_attention_encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def _create_diag_mask(self, size: int) -> torch.Tensor:
        mask = torch.zeros((size, size), dtype=torch.bool)
        mask.fill_diagonal_(True)
        return mask

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        n_sats = residuals.size(0)
        mask = self._create_diag_mask(n_sats).to(residuals.device)

        x = self.embed(residuals.unsqueeze(-1))  # (N, N, D)

        attn_mask = mask                         # True = ignore
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)

        # Aggregate per satellite (rows)
        satellite_features = x.mean(dim=1)       # (N, D)

        # One output per satellite
        return self.out(satellite_features)      # (N, output_dim)


class Gnss_single_epoch_net(nn.Module):

    # features = [
    #     'Cn0DbHz_linear',          # signal strength
    #     'sin_elevation',           # sat geometry
    #     'cos_elevation',           # sat geometry
    #     'sin_azimuth',             # sat geometry
    #     'cos_azimuth',             # sat geometry
    #     'residual',                # pseudorange residual (from WLS)
    #     'rate_residual',
    #     'Cn0DbHz_diff',            # signal strength diff from calibrated based on elevation angle
    #     'pr_baseline_weight',
    #     'prr_baseline_weight',
    #     'doppler_residual',
    #     'MultipathIndicator'
    # ]

    # _feature_standardise_map = torch.tensor([
    #     True,   # Cn0DbHz_linear
    #     True,   # sin_elevation
    #     True,   # cos_elevation
    #     True,   # sin_azimuth
    #     True,   # cos_azimuth
    #     True,   # residual
    #     True,   # rate_residual
    #     True,   # Cn0DbHz_diff
    #     True,   # pr_baseline_weight
    #     True,   # prr_baseline_weight
    #     True,   # doppler_residual
    #     False,  # MultipathIndicator
    # ], dtype=torch.bool)

    features = [
        'Cn0DbHz_linear',          # signal strength
        'sin_elevation',           # sat geometry
        'cos_elevation',           # sat geometry
        'sin_azimuth',             # sat geometry
        'cos_azimuth',             # sat geometry
        'residual',                # pseudorange residual (from WLS)
        'rate_residual',
        'doppler_residual',
        'cn0_over_sine',
        'cn0_stability',           # signal strength stability over time
        'pr_stability',            # pseudorange stability over time
        'prr_stability',           # pseudorange rate stability over time
        # State indicators
        'code_lock',
        'bit_sync',
        'frame_sync',
        'time_decoded',
        'ambiguity_resolved',
        # ADR state indicators
        'valid',
        'reset',
        'cycle_slip',
        'half_cycle_resolved',
        'half_cycle_reported',
    ]

    _feature_standardise_map = torch.tensor([
        True,   # Cn0DbHz_linear
        True,   # sin_elevation
        True,   # cos_elevation
        True,   # sin_azimuth
        True,   # cos_azimuth
        True,   # residual
        True,   # rate_residual
        True,   # doppler_residual
        True,   # cn0_over_sine
        True,   # cn0_stability
        True,   # pr_stability
        True,   # prr_stability
        # State indicators
        False,  # code_lock
        False,  # bit_sync
        False,  # frame_sync
        False,  # time_decoded
        False,  # ambiguity_resolved
        # ADR state indicators
        False,  # valid
        False,  # reset
        False,  # cycle_slip
        False,  # half_cycle_resolved
        False,  # half_cycle_reported
    ], dtype=torch.bool)
    
    """
    Combines the per-satellite and multi-satellite encoders
    and produces pseudorange weight and error predictions.
    """
    def __init__(
        self,
        feat_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        per_sat_hidden: int = 256,
        per_sat_emb_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        pairwise_attn_hidden: int = 128,
        pairwise_attn_output: int = 128,
        joint_hidden: int = 512,
        dropout: float = 0.1,
        need_standardising: torch.Tensor = None,
        require_standardisation: bool = True,
    ):
        super().__init__()

        if need_standardising is None:
            need_standardising = Gnss_single_epoch_net._feature_standardise_map

        self.require_standardisation = require_standardisation
        self.standardiser_ = Standardiser(mean, std, need_standardising)

        self.single_sat_encoder = Single_sat_encoder(
            feat_dim=feat_dim,
            hidden_dim=per_sat_hidden,
            emb_dim=per_sat_emb_dim,
            dropout=dropout,
        )

        self.multi_sat_encoder = Multi_sat_encoder(
            feat_dim=feat_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
        )

        self.pr_pairwise_attn_encoder = Pairwise_attention_encoder(
            hidden_dim=pairwise_attn_hidden,
            output_dim=pairwise_attn_output,
        )

        self.prr_pairwise_attn_encoder = Pairwise_attention_encoder(
            hidden_dim=pairwise_attn_hidden,
            output_dim=pairwise_attn_output,
        )

        joint_input_dim = per_sat_emb_dim + lstm_hidden + 2*pairwise_attn_output

        self.joint_mlp_ = nn.Sequential(
            nn.Linear(joint_input_dim, joint_hidden),
            nn.ReLU(inplace=True),
        )

        # Weight head MLP
        self.pr_weight_head_ = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(joint_hidden // 2),
            nn.Linear(joint_hidden // 2, 1)  # final sigmoid applied in forward
        )

        # Error head MLP
        self.pr_error_head_ = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden // 2, 1)  # raw prediction, no activation
        )

        # Weight head MLP
        self.prr_weight_head_ = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(joint_hidden // 2),
            nn.Linear(joint_hidden // 2, 1)  # final sigmoid applied in forward
        )

        # Error head MLP
        self.prr_error_head_ = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(joint_hidden // 2, 1)  # raw prediction, no activation
        )

    def forward(self, sats: torch.Tensor, pr_residual_matrix: torch.Tensor, prr_residual_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sats.dim() != 2:
            raise ValueError("sats must be (num_sats, feat_dim)")

        if self.require_standardisation:
            sats = self.standardiser_(sats)
        i = Gnss_multi_epoch_net.features.index("residual")
        j = Gnss_multi_epoch_net.features.index("rate_residual")
        pr_residual_matrix = pr_residual_matrix - self.standardiser_.mean[i]
        pr_residual_matrix = pr_residual_matrix / self.standardiser_.std[i]
        prr_residual_matrix = prr_residual_matrix - self.standardiser_.mean[j]
        prr_residual_matrix = prr_residual_matrix / self.standardiser_.std[j]

        per_sat_emb = self.single_sat_encoder(sats)
        lstm_emb = self.multi_sat_encoder(sats)

        pr_pairwise_emb = self.pr_pairwise_attn_encoder(pr_residual_matrix)
        prr_pairwise_emb = self.prr_pairwise_attn_encoder(prr_residual_matrix)

        joint = torch.cat([per_sat_emb, lstm_emb, pr_pairwise_emb, prr_pairwise_emb], dim=1)
        joint_feat = self.joint_mlp_(joint)

        pr_weight_logits = self.pr_weight_head_(joint_feat)
        pr_mean = pr_weight_logits.mean(dim=0, keepdim=True)
        pr_std = pr_weight_logits.std(dim=0, keepdim=True) + 1e-6
        pr_weight_logits = (pr_weight_logits - pr_mean) / pr_std
        pr_weights = torch.sigmoid(pr_weight_logits)
        pr_errors = self.pr_error_head_(joint_feat)

        prr_weight_logits = self.prr_weight_head_(joint_feat)
        prr_mean = prr_weight_logits.mean(dim=0, keepdim=True)
        prr_std = prr_weight_logits.std(dim=0, keepdim=True) + 1e-6
        prr_weight_logits = (prr_weight_logits - prr_mean) / prr_std
        prr_weights = torch.sigmoid(prr_weight_logits)
        # prr_errors = self.prr_error_head_(joint_feat)

        # return pr_weights, pr_errors, prr_weights, prr_errors
        # return pr_weights, prr_weights
        
        # pr_weights = torch.diag_embed(pr_weights.squeeze(1))
        # pr_errors = pr_errors.T.squeeze(0)
        # prr_weights = torch.diag_embed(prr_weights.squeeze(1))

        return pr_weights, pr_errors, prr_weights
    
class Gnss_multi_epoch_net(Gnss_single_epoch_net): # Technically incorrect, but can reuse code

    features = [
        'Cn0DbHz_linear',          # signal strength
        'sin_elevation',           # sat geometry
        'cos_elevation',           # sat geometry
        'sin_azimuth',             # sat geometry
        'cos_azimuth',             # sat geometry
        'residual',                # pseudorange residual (from WLS)
        'rate_residual',
        'doppler_residual',
        'cn0_over_sine',
        'cn0_stability',           # signal strength stability over time
        'pr_stability',            # pseudorange stability over time
        'prr_stability',           # pseudorange rate stability over time
        # State indicators
        'code_lock',
        'bit_sync',
        'frame_sync',
        'time_decoded',
        'ambiguity_resolved',
        # ADR state indicators
        'valid',
        'reset',
        'cycle_slip',
        'half_cycle_resolved',
        'half_cycle_reported',
    ]

    _feature_standardise_map = torch.tensor([
        True,   # Cn0DbHz_linear
        True,   # sin_elevation
        True,   # cos_elevation
        True,   # sin_azimuth
        True,   # cos_azimuth
        True,   # residual
        True,   # rate_residual
        True,   # doppler_residual
        True,   # cn0_over_sine
        True,   # cn0_stability
        True,   # pr_stability
        True,   # prr_stability
        # State indicators
        False,  # code_lock
        False,  # bit_sync
        False,  # frame_sync
        False,  # time_decoded
        False,  # ambiguity_resolved
        # ADR state indicators
        False,  # valid
        False,  # reset
        False,  # cycle_slip
        False,  # half_cycle_resolved
        False,  # half_cycle_reported
    ], dtype=torch.bool)

    """
    Combines the per-satellite temporal encoder and multi-satellite encoder
    and produces pseudorange weight and error predictions.
    """
    def __init__(
        self,
        feat_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        temporal_emb_dim: int = 256,
        per_sat_lstm_hidden: int = 256,
        per_sat_lstm_layers: int = 1,
        per_sat_emb_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        joint_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super(Gnss_multi_epoch_net, self).__init__(
            feat_dim=temporal_emb_dim,
            mean=mean,
            std=std,
            per_sat_hidden=per_sat_lstm_hidden,
            per_sat_emb_dim=per_sat_emb_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            joint_hidden=joint_hidden,
            dropout=dropout,
            require_standardisation=False,
            need_standardising=Gnss_multi_epoch_net._feature_standardise_map,
        )

        self.temporal_sat_encoder = Temporal_sat_encoder(
            feat_dim=feat_dim,
            lstm_hidden=per_sat_lstm_hidden,
            lstm_layers=per_sat_lstm_layers,
            emb_dim=per_sat_emb_dim,
            dropout=dropout,
        )

    def forward(self, sats: torch.Tensor, lengths: torch.Tensor, pr_residual_matrix: torch.Tensor, prr_residual_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sats.dim() != 3:
            raise ValueError("sats must be (num_sats, time_steps, feat_dim)")

        num_sats, _, _ = sats.shape

        # Flatten sats for standardisation
        sats_flat = sats.view(-1, sats.size(-1))
        sats_flat = self.standardiser_(sats_flat)
        sats = sats_flat.view(num_sats, -1, sats.size(-1))

        per_sat_emb = self.temporal_sat_encoder(sats, lengths)
        return super().forward(sats=per_sat_emb, pr_residual_matrix=pr_residual_matrix, prr_residual_matrix=prr_residual_matrix)