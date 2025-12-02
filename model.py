import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.max(torch.abs(x))
    

class Standardiser(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    

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
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm_.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, sats: torch.Tensor) -> torch.Tensor:
        # sats: (num_sats, feat_dim)
        seq = sats.unsqueeze(1)  # (num_sats, 1, feat_dim)
        lstm_out, _ = self.lstm_(seq)
        return lstm_out.squeeze(1)  # (num_sats, lstm_hidden)


class Gnss_weighting_visibility_net(nn.Module):

    features = [
        'Cn0DbHz',                 # signal strength
        'sin_elevation',           # sat geometry
        'cos_elevation',           # sat geometry
        'sin_azimuth',             # sat geometry
        'cos_azimuth',             # sat geometry
        'los_x',
        'los_y',
        'los_z',
        'residual',                # pseudorange residual (from WLS)
        'rate_residual',
        'Cn0DbHz_diff',            # signal strength diff from calibrated based on elevation angle
        'pr_baseline_weight',
        'prr_baseline_weight',
        'doppler_residual',
        'MultipathIndicator'
    ]
    
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
        joint_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.standardiser_ = Standardiser(mean, std)

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

        joint_input_dim = per_sat_emb_dim + lstm_hidden

        self.joint_mlp_ = nn.Sequential(
            nn.Linear(joint_input_dim, joint_hidden),
            nn.ReLU(inplace=True),
        )

        # Weight head MLP
        self.pr_weight_head_ = nn.Sequential(
            nn.Linear(joint_hidden, joint_hidden),
            nn.ReLU(inplace=True),
            Normalise(),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            Normalise(),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.ReLU(inplace=True),
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

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sats: torch.Tensor):
        if sats.dim() != 2:
            raise ValueError("sats must be (num_sats, feat_dim)")

        sats = self.standardiser_(sats)

        per_sat_emb = self.single_sat_encoder(sats)
        lstm_emb = self.multi_sat_encoder(sats)

        joint = torch.cat([per_sat_emb, lstm_emb], dim=1)
        joint_feat = self.joint_mlp_(joint)

        pr_weight_logits = self.pr_weight_head_(joint_feat)
        pr_weights = torch.sigmoid(pr_weight_logits)
        pr_errors = self.pr_error_head_(joint_feat)

        prr_weight_logits = self.prr_weight_head_(joint_feat)
        prr_weights = torch.sigmoid(prr_weight_logits)
        # prr_errors = self.prr_error_head_(joint_feat)

        # return pr_weights, pr_errors, prr_weights, prr_errors
        # return pr_weights, prr_weights
        return pr_weights, pr_errors, prr_weights