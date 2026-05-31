import torch
import torch.nn as nn
import math

from model.daycent import AttentionPooling, month_day_ranges
from model.registry import register_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@register_model("transformer")
class DayCentTransformer(nn.Module):
    def __init__(self, input_dim, init_dim, year_dim, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1, **kwargs):
        super().__init__()

        # 1. Input Projections
        # Projects daily weather/management features to d_model space
        self.daily_proj = nn.Linear(input_dim, d_model)
        
        # Projects static site info + year encoding + previous state to d_model space
        # We will use this to "contextualize" the daily data
        self.ctx_proj = nn.Linear(init_dim + year_dim + 1, d_model)

        # 2. Transformer Encoder
        # batch_first=True is crucial because data is (B, 365, F)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Heads (Keep your existing logic, it's sound)
        
        # SOMSC Heads (Month-specific)
        # We use a shared attention mechanism for efficiency, but specific linear heads
        self.somsc_attn = AttentionPooling(d_model, d_model)
        self.somsc_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + 1, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(12)
        ])
        
        # Yield Head
        self.yield_attn = AttentionPooling(d_model, d_model)
        self.yield_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self._init_somsc_heads()

    def _init_somsc_heads(self):
        for head in self.somsc_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, batch):
        seq = batch["sequence"]             # (B, 365, Input_F)
        init_cond = batch["init_cond"]      # (B, Init_F)
        year_enc = batch["year_enc"]        # (B, Year_F)
        harvest_mask = batch["harvest_mask"]# (B, 365)
        prev_somsc = batch["prev_somsc_state"].unsqueeze(-1) # (B, 1)

        # --- Feature Engineering ---
        
        # 1. Prepare Daily Embeddings
        x_daily = self.daily_proj(seq) # (B, 365, d_model)
        
        # 2. Prepare Context Embedding (Static + State)
        # Combine static info and previous state into one vector
        context_vec = torch.cat([init_cond, year_enc, prev_somsc], dim=-1)
        x_ctx = self.ctx_proj(context_vec) # (B, d_model)
        
        # 3. Fuse Context into Daily
        # Strategy: Add static context to every time step
        # This is often better than concatenation for Transformers as it preserves dimension
        x = x_daily + x_ctx.unsqueeze(1)
        
        # 4. Positional Encoding & Transformer
        x = self.pos_encoder(x)
        h = self.transformer_encoder(x) # (B, 365, d_model)

        # 1. SOMSC Prediction
        somsc_preds = []
        somsc_deltas = []
        somsc_attns = []
        ranges = month_day_ranges() # Your existing utility function
        
        current_val = prev_somsc.squeeze(-1)

        for m, (start, end) in enumerate(ranges):
            # Mask: Only attend to days up to the end of the current month
            # (Causal masking for the months)
            mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
            mask[:, :end] = True
            
            # Pool the transformer output
            pooled, attn = self.somsc_attn(h, mask=mask)
            
            # Predict DELTA
            head_input = torch.cat([pooled, current_val.unsqueeze(-1)], dim=-1)
            delta = self.somsc_heads[m](head_input).squeeze(-1)
            
            current_val = current_val + delta
            somsc_preds.append(current_val)
            somsc_deltas.append(delta)
            somsc_attns.append(attn)

        somsc_preds = torch.stack(somsc_preds, dim=1) # (B, 12)
        somsc_deltas = torch.stack(somsc_deltas, dim=1) # (B, 12)

        # 2. Yield Prediction
        yield_repr, yield_attn = self.yield_attn(h, mask=harvest_mask)
        yield_pred = self.yield_head(yield_repr).squeeze(-1)

        return {
            "somsc_pred": somsc_preds,
            "somsc_delta_pred": somsc_deltas,
            "yield_pred": yield_pred,
            "somsc_attn": somsc_attns,
            "yield_attn": yield_attn
        }
