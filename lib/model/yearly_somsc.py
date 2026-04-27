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
