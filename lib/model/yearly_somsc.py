import torch
from torch import nn

from model.registry import register_model


@register_model("yearly_somsc")
class YearlySOMSCModel(nn.Module):
    """
    LSTM-based yearly model that predicts December SOMSC from monthly aggregates.
    """

    def __init__(
        self,
        input_dim,
        init_dim,
        year_dim,
        hidden_dim=128,
        latent_dim=32,
        lstm_layers=1,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()

        static_dim = init_dim + year_dim + 1

        self.monthly_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(static_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=latent_dim * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        seq = batch["sequence"]
        init_cond = batch["init_cond"]
        year_enc = batch["year_enc"]
        prev_somsc = batch["prev_somsc_state"]

        monthly_latent = self.monthly_proj(seq)
        static_context = torch.cat(
            [init_cond, year_enc, prev_somsc.unsqueeze(-1)],
            dim=-1,
        )
        context_latent = self.context_proj(static_context)
        repeated_context = context_latent.unsqueeze(1).expand(-1, seq.size(1), -1)

        lstm_input = torch.cat([monthly_latent, repeated_context], dim=-1)
        monthly_hidden, _ = self.lstm(lstm_input)
        final_hidden = monthly_hidden[:, -1]

        somsc_delta = self.output_head(final_hidden).squeeze(-1)
        somsc_pred = prev_somsc + somsc_delta

        return {
            "somsc_pred": somsc_pred,
            "somsc_attn": None,
        }


@register_model("yearly_somsc_state")
class YearlySOMSCStateModel(nn.Module):
    """
    LSTM-based yearly state-space model for December SOC pool updates.

    The model consumes the previous raw pool state as the conserved state and
    the normalized previous pool state as context.  It predicts raw pool deltas
    in the order provided by the dataset:
    [som1c_soil, som2c_soil, som3c, som2c_surface].
    """

    def __init__(
        self,
        input_dim,
        init_dim,
        year_dim,
        state_dim=4,
        hidden_dim=128,
        latent_dim=32,
        lstm_layers=1,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()

        static_dim = init_dim + year_dim + state_dim
        self.state_dim = state_dim

        self.monthly_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(static_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=latent_dim * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, batch):
        seq = batch["sequence"]
        init_cond = batch["init_cond"]
        year_enc = batch["year_enc"]
        prev_state_raw = batch["prev_soc_state_raw"]
        prev_state_norm = batch["prev_soc_state_norm"]

        monthly_latent = self.monthly_proj(seq)
        static_context = torch.cat(
            [init_cond, year_enc, prev_state_norm],
            dim=-1,
        )
        context_latent = self.context_proj(static_context)
        repeated_context = context_latent.unsqueeze(1).expand(-1, seq.size(1), -1)

        lstm_input = torch.cat([monthly_latent, repeated_context], dim=-1)
        monthly_hidden, _ = self.lstm(lstm_input)
        final_hidden = monthly_hidden[:, -1]

        soc_state_delta_pred = self.output_head(final_hidden)
        soc_state_pred = prev_state_raw + soc_state_delta_pred
        somsc_pred = soc_state_pred[:, :3].sum(dim=-1)

        return {
            "soc_state_delta_pred": soc_state_delta_pred,
            "soc_state_pred": soc_state_pred,
            "somsc_pred": somsc_pred,
            "somsc_attn": None,
        }
