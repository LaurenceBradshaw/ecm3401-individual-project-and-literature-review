import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from internals.preprocessing.hash_file import preprocessing_artifacts_path

torch.set_default_dtype(torch.float64)
torch.manual_seed(42 * 42) # The meaning of life, the universe, and everything GNSS squared for good measure

def _load_string_list(file_path: str) -> list[str]:
    """
    Loads a list of strings from a text file, where each line corresponds to one string.

    Parameters
    ----------
    file_path: str
        The path to the text file containing the list of strings.

    Returns
    -------
    list[str]
        The list of strings loaded from the file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}, please run the preprocessing script.")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
    
def get_test_drives(base_path: str) -> list[str]:
    """
    Read the list of test drives from the preprocessing artifacts. 
    The test drives are stored in a text file where each line is a relative path to a drive directory. 
    Prepend the base path to get the full paths to the test drives.

    Parameters
    ----------
    base_path: str
        The base directory containing drive data subdirectories.

    Returns
    -------
    list[str]
        A list of full paths to the test drives.
    """
    test_drives = _load_string_list(os.path.join(preprocessing_artifacts_path(), "test_drives.txt"))
    # add base path back to test drives
    test_drives = [os.path.join(base_path, d) for d in test_drives]
    return test_drives

def get_training_drives(base_path: str) -> list[str]:
    """
    Get the list of training drives by excluding any drives that are in the same directory as 
    the test drives, to prevent data leakage.

    Parameters
    ----------
    base_path: str
        The base directory containing drive data subdirectories.

    Returns
    -------
    list[str]
        A list of full paths to the training drives.
    """
    test_drives = set(get_test_drives(base_path))
    test_drives_no_phone = set(os.path.dirname(d) for d in test_drives)
    all_drives = [
        os.path.join(base_path, d, p)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
        for p in os.listdir(os.path.join(base_path, d))
        if os.path.isdir(os.path.join(base_path, d, p))
    ]
    # Ensure no drive from the same directory as a test drive is included in training drives, to prevent data leakage
    # since the phones in the same directory have been on the same routes and have the same satellite visibility patterns.
    training_drives = [d for d in all_drives if os.path.dirname(d) not in test_drives_no_phone]

    return training_drives

class Standardiser(nn.Module):
    """
    Standardises the input features using the provided mean and std, but only for the features that require standardisation.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, need_standardising: torch.Tensor):
        """
        Parameters
        ----------
        mean: torch.Tensor
            The mean values for each feature, used for standardisation.
        std: torch.Tensor
            The standard deviation values for each feature, used for standardisation.
        need_standardising: torch.Tensor
            A boolean tensor indicating which features require standardisation (True for standardise, False to leave unchanged).
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("need_standardising", need_standardising)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that standardises the input features.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (num_sats, feat_dim) containing the features for each satellite.

        Returns
        -------
        torch.Tensor
            The output tensor with the same shape as input, where the features that require standardisation have been standardised.
        """
        x_std = (x - self.mean) / self.std
        return torch.where(self.need_standardising, x_std, x)


class Residual_block(nn.Module):
    """
    A residual block with two linear layers, ReLU activations, optional dropout,
    and LayerNorm applied after the skip connection.
    Input and output dims are both `dim`, so no projection is needed.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        dim: int
            The input and output dimension of the residual block.
        dropout: float, optional
            The dropout rate to be applied after the first linear layer, by default 0.1.
        """
        super().__init__()
        self.block_ = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
        )
        self.norm_ = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (dim,).
        """
        return self.norm_(x + self.block_(x))


class Single_sat_encoder(nn.Module):
    """
    Encodes each satellite independently using a small MLP with a residual
    skip connection from input to output (projected if dims differ).
    """
    def __init__(self, feat_dim: int, hidden_dim: int, emb_dim: int, dropout: float):
        """
        Parameters
        ----------
        feat_dim: int
            The dimension of the input features for each satellite.
        hidden_dim: int
            The dimension of the hidden layer in the MLP.
        emb_dim: int
            The dimension of the output embedding for each satellite.
        dropout: float
            The dropout rate to be applied in the MLP.
        """
        super().__init__()
        self.net_ = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LeakyReLU(inplace=True),
        )
        # Project input to emb_dim for the residual if dims differ
        self.residual_proj_ = (
            nn.Linear(feat_dim, emb_dim, bias=False)
            if feat_dim != emb_dim else nn.Identity()
        )
        self.norm_ = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the single satellite encoder.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (num_sats, feat_dim) containing the features for each satellite.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (num_sats, emb_dim) containing the encoded features for each satellite.
        """
        return self.norm_(self.net_(x) + self.residual_proj_(x))


class Multi_sat_encoder(nn.Module):
    """
    Encodes the entire satellite set using an LSTM to capture live sky conditions.
    """
    def __init__(self, feat_dim: int, lstm_hidden: int, lstm_layers: int):
        """
        Parameters
        ----------
        feat_dim: int
            The dimension of the input features for each satellite.
        lstm_hidden: int
            The hidden dimension of the LSTM.
        lstm_layers: int
            The number of layers in the LSTM.
        """
        super().__init__()
        self.lstm_ = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=False,
        )

    def forward(self, sats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-satellite encoder.

        Parameters
        ----------
        sats: torch.Tensor
            The input tensor of shape (num_sats, feat_dim) containing the features for each satellite.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (num_sats, lstm_hidden) containing the encoded features for each satellite.
        """
        seq = sats.unsqueeze(1) # (num_sats, 1, feat_dim)
        lstm_out, _ = self.lstm_(seq)
        return lstm_out.squeeze(1) # (num_sats, lstm_hidden)


class Temporal_sat_encoder(nn.Module):
    """
    Encodes the per-satellite time sequence into a fixed-size vector.
    Input shape:  (num_sats, time_steps, feat_dim)
    Output shape: (num_sats, emb_dim)
    """
    def __init__(self, feat_dim: int, lstm_hidden: int, lstm_layers: int, emb_dim: int, dropout: float):
        """
        Parameters
        ----------
        feat_dim: int
            The dimension of the input features for each satellite at each time step.
        lstm_hidden: int
            The hidden dimension of the LSTM.
        lstm_layers: int
            The number of layers in the LSTM.
        emb_dim: int
            The dimension of the output embedding for each satellite.
        dropout: float
            The dropout rate to be applied in the LSTM.
        """
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
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Skip connection: lstm_hidden -> emb_dim (bias=False keeps it a pure linear map)
        self.skip_ = nn.Linear(lstm_hidden, emb_dim, bias=False)
        self.norm_ = nn.LayerNorm(emb_dim)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal satellite encoder.

        Parameters
        ----------
        seq: torch.Tensor
            The input tensor of shape (num_sats, time_steps, feat_dim) containing the features for each satellite at each time step.
        lengths: torch.Tensor
            The tensor of shape (num_sats,) containing the valid sequence lengths for each satellite.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (num_sats, emb_dim) containing the encoded features for each satellite.
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm_(packed)

        # Final hidden state from last layer: (num_sats, lstm_hidden)
        last_hidden = h_n[-1]

        return self.norm_(self.proj_(last_hidden) + self.skip_(last_hidden))


class Pairwise_attention_encoder(nn.Module):
    """
    Encodes pairwise relationships between satellites using attention mechanisms.
    """
    def __init__(self, hidden_dim, output_dim):
        """
        Parameters
        ----------
        hidden_dim: int
            The dimension of the hidden states in the attention mechanism.
        output_dim: int
            The dimension of the output tensor.
        """
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
        """
        Creates a diagonal mask for the attention mechanism.

        Parameters
        ----------
        size: int
            The size of the square mask to be created.
        """
        mask = torch.zeros((size, size), dtype=torch.bool)
        mask.fill_diagonal_(True)
        return mask

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pairwise attention encoder.

        Parameters
        ----------
        residuals: torch.Tensor
            The input tensor of shape (num_sats, num_sats) containing the residuals between satellites.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (num_sats, output_dim) containing the encoded pairwise relationships.
        """
        n_sats = residuals.size(0)
        mask = self._create_diag_mask(n_sats).to(residuals.device)

        x = self.embed(residuals.unsqueeze(-1)) # (N, N, D)

        attn_mask = mask # True = ignore
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)

        # Aggregate per satellite (rows)
        satellite_features = x.mean(dim=1) # (N, D)

        # One output per satellite
        return self.out(satellite_features) # (N, output_dim)


class Gnss_single_epoch_net(nn.Module):
    features = [
        'Cn0DbHz_linear',          # signal strength
        'sin_elevation',           # sat geometry
        'cos_elevation',           # sat geometry
        'sin_azimuth',             # sat geometry
        'cos_azimuth',             # sat geometry
        'residual',                # pseudorange residual (from WLS)
        'rate_residual',           # pseudorange rate residual (from WLS)
        'doppler_residual',        # doppler residual. estimated by doppler shift
        'cn0_stability',           # signal strength stability over time
        'pr_stability',            # pseudorange stability over time
        'prr_stability',           # pseudorange rate stability over time
        'adr_td_residual',         # time-differenced ADR residual. estimated from the change in ADR between epochs compared to prr
        'adr_td_valid',            # whether the ADR time-differenced residual is valid
        # ADR state indicators
        'valid',
        'reset',
        'cycle_slip',
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
        True,   # cn0_stability
        True,   # pr_stability
        True,   # prr_stability
        True,   # adr_td_residual
        False,  # adr_td_valid
        # ADR state indicators
        False,  # valid
        False,  # reset
        False,  # cycle_slip
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
        per_sat_hidden: int = 512,
        per_sat_emb_dim: int = 512,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        pairwise_attn_hidden: int = 128,
        pairwise_attn_output: int = 128,
        joint_hidden: int = 512,
        dropout: float = 0.1,
        need_standardising: torch.Tensor = None,
        require_standardisation: bool = True,
    ):
        """
        Parameters
        ----------
        feat_dim: int
            The dimension of the input features.
        mean: torch.Tensor
            The mean values for feature standardisation.
        std: torch.Tensor
            The standard deviation values for feature standardisation.
        per_sat_hidden: int
            The dimension of the hidden states in the per-satellite encoder.
        per_sat_emb_dim: int
            The dimension of the embedding space in the per-satellite encoder.
        lstm_hidden: int
            The dimension of the hidden states in the LSTM.
        lstm_layers: int
            The number of layers in the LSTM.
        pairwise_attn_hidden: int
            The dimension of the hidden states in the pairwise attention mechanism.
        pairwise_attn_output: int
            The dimension of the output tensor from the pairwise attention mechanism.
        joint_hidden: int
            The dimension of the hidden states in the joint MLP.
        dropout: float
            The dropout rate for regularization.
        need_standardising: torch.Tensor
            A boolean tensor indicating which features need standardisation.
        require_standardisation: bool
            Whether to apply standardisation to the input features.
        """
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

        joint_input_dim = per_sat_emb_dim + lstm_hidden + 2 * pairwise_attn_output

        self.joint_mlp_ = nn.Sequential(
            nn.Linear(joint_input_dim, joint_hidden),
            nn.LeakyReLU(inplace=True),
        )

        # PR weight head
        # Two residual blocks keep gradients healthy through the deeper path;
        # the final Linear + softplus is applied in forward().
        self.pr_weight_head_ = nn.Sequential(
            Residual_block(joint_hidden, dropout=dropout),
            Residual_block(joint_hidden, dropout=dropout),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(joint_hidden // 2),
            nn.Linear(joint_hidden // 2, 1),   # softplus applied in forward
        )

        # PR error head
        self.pr_error_head_ = nn.Sequential(
            Residual_block(joint_hidden, dropout=dropout),
            Residual_block(joint_hidden, dropout=dropout),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(joint_hidden // 2, 1),   # raw prediction, no activation
        )

        # PRR weight head
        self.prr_weight_head_ = nn.Sequential(
            Residual_block(joint_hidden, dropout=dropout),
            Residual_block(joint_hidden, dropout=dropout),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(joint_hidden // 2),
            nn.Linear(joint_hidden // 2, 1),   # softplus applied in forward
        )

        # PRR error head
        self.prr_error_head_ = nn.Sequential(
            Residual_block(joint_hidden, dropout=dropout),
            Residual_block(joint_hidden, dropout=dropout),
            nn.Linear(joint_hidden, joint_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(joint_hidden // 2, 1),   # raw prediction, no activation
        )

    def forward(
        self,
        sats: torch.Tensor,
        pr_residual_matrix: torch.Tensor,
        prr_residual_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNSS single epoch network.

        Parameters
        ----------
        sats: torch.Tensor
            The input tensor of shape (num_sats, feat_dim) containing the satellite features.
        pr_residual_matrix: torch.Tensor
            The input tensor of shape (num_sats, num_sats) containing the PR residuals between satellites.
        prr_residual_matrix: torch.Tensor
            The input tensor of shape (num_sats, num_sats) containing the PRR residuals between satellites.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the weight matrices and error vectors for PR and PRR.
        """
        if sats.dim() != 2:
            raise ValueError("sats must be (num_sats, feat_dim)")

        if self.require_standardisation:
            sats = self.standardiser_(sats)

        i = Gnss_single_epoch_net.features.index("residual")
        j = Gnss_single_epoch_net.features.index("rate_residual")
        pr_residual_matrix = (pr_residual_matrix - self.standardiser_.mean[i]) / self.standardiser_.std[i]
        prr_residual_matrix = (prr_residual_matrix - self.standardiser_.mean[j]) / self.standardiser_.std[j]

        per_sat_emb = self.single_sat_encoder(sats)
        lstm_emb = self.multi_sat_encoder(sats)

        pr_pairwise_emb  = self.pr_pairwise_attn_encoder(pr_residual_matrix)
        prr_pairwise_emb = self.prr_pairwise_attn_encoder(prr_residual_matrix)

        joint = torch.cat([per_sat_emb, lstm_emb, pr_pairwise_emb, prr_pairwise_emb], dim=1)
        joint_feat = self.joint_mlp_(joint)

        pr_weight_logits = self.pr_weight_head_(joint_feat)
        pr_sigma = F.softplus(pr_weight_logits) + 1e-3
        pr_weights = 1.0 / (pr_sigma * pr_sigma)
        pr_weights = pr_weights / pr_weights.mean()
        pr_errors = self.pr_error_head_(joint_feat)

        prr_weight_logits = self.prr_weight_head_(joint_feat)
        prr_sigma = F.softplus(prr_weight_logits) + 1e-3
        prr_weights = 1.0 / (prr_sigma * prr_sigma)
        prr_weights = prr_weights / prr_weights.mean()
        prr_errors = self.prr_error_head_(joint_feat)

        assert not torch.allclose(pr_weights, pr_weights[0]),   "All PR weights are the same, model is not learning"
        assert not torch.allclose(prr_weights, prr_weights[0]), "All PRR weights are the same, model is not learning"

        Wx = torch.diag_embed(pr_weights.squeeze(1))
        pr_errors = torch.clamp(pr_errors.T.squeeze(0), -100, 100)
        Wv = torch.diag_embed(prr_weights.squeeze(1))
        prr_errors = torch.clamp(prr_errors.T.squeeze(0), -10, 10)

        return Wx, pr_errors, Wv, prr_errors


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
        'cn0_stability',           # signal strength stability over time
        'pr_stability',            # pseudorange stability over time
        'prr_stability',           # pseudorange rate stability over time
        'adr_td_residual',         # time-differenced ADR residual
        'adr_td_valid',            # whether the ADR time-differenced residual is valid
        # ADR state indicators
        'valid',
        'reset',
        'cycle_slip',
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
        True,   # cn0_stability
        True,   # pr_stability
        True,   # prr_stability
        True,   # adr_td_residual
        False,  # adr_td_valid
        # ADR state indicators
        False,  # valid
        False,  # reset
        False,  # cycle_slip
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
        per_sat_lstm_hidden: int = 256,
        per_sat_lstm_layers: int = 1,
        per_sat_emb_dim: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        joint_hidden: int = 512,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        feat_dim: int
            The dimension of the input features.
        mean: torch.Tensor
            The mean values for feature standardisation.
        std: torch.Tensor
            The standard deviation values for feature standardisation.
        per_sat_lstm_hidden: int
            The dimension of the hidden states in the per-satellite LSTM.
        per_sat_lstm_layers: int
            The number of layers in the per-satellite LSTM.
        per_sat_emb_dim: int
            The dimension of the embedding space in the per-satellite encoder.
        lstm_hidden: int
            The dimension of the hidden states in the LSTM.
        lstm_layers: int
            The number of layers in the LSTM.
        joint_hidden: int
            The dimension of the hidden states in the joint MLP.
        dropout: float
            The dropout rate for regularization.
        """
        super(Gnss_multi_epoch_net, self).__init__(
            feat_dim=per_sat_emb_dim,
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

        self.current_feat_proj_ = nn.Sequential(
            nn.Linear(per_sat_emb_dim + feat_dim, per_sat_emb_dim),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(per_sat_emb_dim),
        )

    def forward(
        self,
        sats: torch.Tensor,
        lengths: torch.Tensor,
        pr_residual_matrix: torch.Tensor,
        prr_residual_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNSS multi-epoch network.

        Parameters
        ----------
        sats: torch.Tensor
            The input tensor of shape (num_sats, time_steps, feat_dim) containing the satellite features.
        lengths: torch.Tensor
            The input tensor of shape (num_sats,) containing the lengths of each satellite's sequence.
        pr_residual_matrix: torch.Tensor
            The input tensor of shape (num_sats, num_sats) containing the PR residuals between satellites.
        prr_residual_matrix: torch.Tensor
            The input tensor of shape (num_sats, num_sats) containing the PRR residuals between satellites.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the weight matrices and error vectors for PR and PRR.
        """
        if sats.dim() != 3:
            raise ValueError("sats must be (num_sats, time_steps, feat_dim)")

        num_sats, _, _ = sats.shape

        # Flatten sats for standardisation
        sats_flat = sats.view(-1, sats.size(-1))
        sats_flat = self.standardiser_(sats_flat)
        sats = sats_flat.view(num_sats, -1, sats.size(-1))

        # Current epoch features (last timestep) - these must not get lost in the LSTM
        current_feats = sats[:, -1, :]

        per_sat_emb = self.temporal_sat_encoder(sats, lengths)

        # Concatenate current epoch features directly so instantaneous signals
        # like elevation cannot be overridden by historical LSTM representation
        per_sat_emb = torch.cat([per_sat_emb, current_feats], dim=1)
        per_sat_emb = self.current_feat_proj_(per_sat_emb)

        return super().forward(
            sats=per_sat_emb,
            pr_residual_matrix=pr_residual_matrix,
            prr_residual_matrix=prr_residual_matrix,
        )