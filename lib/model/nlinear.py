"""
NLinear model for DayCent.

Based on "Are Transformers Effective for Time Series Forecasting?"
https://arxiv.org/abs/2205.13504

NLinear applies a simple normalization trick:
1. Subtract the last value of the input sequence (distribution shift handling)
2. Apply a linear transformation
3. Add the subtracted value back

Adapted for DayCent's multi-task setup (SOMSC + Yield prediction).
"""
import torch
import torch.nn as nn

from model.registry import register_model


@register_model("nlinear_simple")
class NLinearSimple(nn.Module):
    """
    NLinear variant with non-linearities for stable training.
    
    Based on the original NLinear but adds:
    - Hidden layer with GELU activation for stable gradients
    - LayerNorm to handle large value ranges (SOMSC ~1000s)
    - Residual connection from prev_somsc for SOMSC prediction
    """
    
    def __init__(self, input_dim, init_dim, year_dim, hidden_dim=256, **kwargs):
        super().__init__()
        
        self.seq_len = 365
        self.pred_len = 12
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Flatten dimension
        self.flatten_dim = self.seq_len * input_dim
        
        # Static feature dimension
        static_dim = init_dim + year_dim + 1  # +1 for prev_somsc
        
        # Sequence encoder with non-linearity
        self.seq_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Static encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # SOMSC head: predicts deltas from prev_somsc
        self.somsc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pred_len)
        )
        
        # Yield head
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, batch):
        seq = batch["sequence"]             # (B, 365, F)
        init_cond = batch["init_cond"]      # (B, I)
        year_enc = batch["year_enc"]        # (B, Y)
        prev_somsc = batch["prev_somsc_state"]  # (B,)
        
        B = seq.size(0)
        
        # Flatten and encode sequence
        x_flat = seq.view(B, -1)  # (B, 365 * F)
        seq_feat = self.seq_encoder(x_flat)  # (B, hidden_dim)
        
        # Static features
        static = torch.cat([
            init_cond, 
            year_enc, 
            prev_somsc.unsqueeze(-1)
        ], dim=-1)
        static_feat = self.static_encoder(static)  # (B, hidden_dim)
        
        # Combined features
        combined = torch.cat([seq_feat, static_feat], dim=-1)  # (B, 2*hidden_dim)
        
        # SOMSC prediction (predict deltas, add to prev_somsc)
        somsc_delta = self.somsc_head(combined)  # (B, 12)
        somsc_preds = prev_somsc.unsqueeze(-1) + somsc_delta
        
        # Yield prediction
        yield_pred = self.yield_head(combined).squeeze(-1)  # (B,)
        
        return {
            "somsc_pred": somsc_preds,
            "yield_pred": yield_pred,
            "somsc_attn": None,
            "yield_attn": None
        }
