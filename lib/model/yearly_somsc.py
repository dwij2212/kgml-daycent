import torch
from torch import nn

from model.registry import register_model


SOC_STATE_COLS = ["som1c_soil", "som2c_soil", "som3c", "som2c_surface"]
SOIL_POOL_COLS = ["som1c_soil", "som2c_soil", "som3c"]
VALID_TARGET_MODES = {"pool_deltas", "somsc_delta"}
VALID_CONTEXT_MODES = {"target_pools", "soil_pools", "full_pools", "somsc"}


def _resolve_pool_indices(pool_cols, state_dim):
    if pool_cols is None:
        pool_cols = SOC_STATE_COLS[:state_dim]

    indices = []
    for col in pool_cols:
        if col not in SOC_STATE_COLS:
            raise ValueError(
                f"Unknown SOC pool column '{col}'. Valid columns: {SOC_STATE_COLS}"
            )
        idx = SOC_STATE_COLS.index(col)
        if idx >= state_dim:
            raise ValueError(
                f"Pool column '{col}' maps to index {idx}, but state_dim={state_dim}."
            )
        indices.append(idx)

    if not indices:
        raise ValueError("At least one target pool column is required.")
    return indices


@register_model("yearly_somsc_state")
class YearlySOMSCStateModel(nn.Module):
    """
    LSTM-based yearly model for December SOC transitions and ablations.

    The default configuration is the 4-state pool model:
    previous normalized pools as context, raw pool deltas as targets, and SOMSC
    derived by summing the three soil pools.  Ablations can instead expose only
    aggregate previous SOMSC as context, predict only the three soil pools, or
    predict one scalar annual SOMSC delta.
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
        target_mode="pool_deltas",
        prev_state_context="target_pools",
        target_pool_cols=None,
        **kwargs,
    ):
        super().__init__()

        if target_mode not in VALID_TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {sorted(VALID_TARGET_MODES)}, "
                f"got {target_mode!r}."
            )
        if prev_state_context not in VALID_CONTEXT_MODES:
            raise ValueError(
                "prev_state_context must be one of "
                f"{sorted(VALID_CONTEXT_MODES)}, got {prev_state_context!r}."
            )

        self.state_dim = state_dim
        self.target_mode = target_mode
        self.prev_state_context = prev_state_context
        self.target_pool_indices = _resolve_pool_indices(target_pool_cols, state_dim)
        self.target_pool_cols = [SOC_STATE_COLS[idx] for idx in self.target_pool_indices]

        if prev_state_context == "target_pools":
            context_pool_indices = self.target_pool_indices
            context_dim = len(context_pool_indices)
        elif prev_state_context == "soil_pools":
            context_pool_indices = [SOC_STATE_COLS.index(col) for col in SOIL_POOL_COLS]
            context_dim = len(context_pool_indices)
        elif prev_state_context == "full_pools":
            context_pool_indices = list(range(state_dim))
            context_dim = state_dim
        else:
            context_pool_indices = []
            context_dim = 1

        self.context_pool_indices = context_pool_indices
        output_dim = 1 if target_mode == "somsc_delta" else len(self.target_pool_indices)
        static_dim = init_dim + year_dim + context_dim

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
            nn.Linear(hidden_dim, output_dim),
        )

    def _previous_state_context(self, batch):
        if self.prev_state_context == "somsc":
            return batch["prev_somsc_norm"].unsqueeze(-1)
        return batch["prev_soc_state_norm"][:, self.context_pool_indices]

    def forward(self, batch):
        seq = batch["sequence"]
        init_cond = batch["init_cond"]
        year_enc = batch["year_enc"]
        prev_state_raw = batch["prev_soc_state_raw"]
        prev_somsc = batch["prev_somsc_state"]

        monthly_latent = self.monthly_proj(seq)
        prev_context = self._previous_state_context(batch)
        static_context = torch.cat(
            [init_cond, year_enc, prev_context],
            dim=-1,
        )
        context_latent = self.context_proj(static_context)
        repeated_context = context_latent.unsqueeze(1).expand(-1, seq.size(1), -1)

        lstm_input = torch.cat([monthly_latent, repeated_context], dim=-1)
        monthly_hidden, _ = self.lstm(lstm_input)
        final_hidden = monthly_hidden[:, -1]

        raw_pred = self.output_head(final_hidden)
        if self.target_mode == "somsc_delta":
            somsc_delta_pred = raw_pred.squeeze(-1)
            somsc_pred = prev_somsc + somsc_delta_pred
            return {
                "somsc_delta_pred": somsc_delta_pred,
                "somsc_pred": somsc_pred,
                "somsc_attn": None,
            }

        target_delta_pred = raw_pred
        soc_state_delta_pred = prev_state_raw.new_zeros(prev_state_raw.shape)
        soc_state_delta_pred[:, self.target_pool_indices] = target_delta_pred
        soc_state_pred = prev_state_raw + soc_state_delta_pred
        somsc_pred = soc_state_pred[:, :3].sum(dim=-1)

        return {
            "soc_state_delta_pred": soc_state_delta_pred,
            "target_soc_state_delta_pred": target_delta_pred,
            "soc_state_pred": soc_state_pred,
            "somsc_pred": somsc_pred,
            "somsc_attn": None,
        }
