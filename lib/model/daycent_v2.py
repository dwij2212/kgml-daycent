"""
DayCent Model v2 — Improved architecture for emulating the DayCent physical model.

Key improvements over v1:
─────────────────────────
1. RICHER DAILY ENCODER: A small MLP with LayerNorm replaces the single Linear,
   so the model can learn non-linear feature interactions (e.g. Tmin×Precip)
   *before* the sequence model sees them.

2. FILM CONDITIONING (Feature-wise Linear Modulation): Instead of naively
   repeating the global (init + year) vector and concatenating it, we learn
   per-feature scale (γ) and shift (β) from the static context. This is the
   standard way to condition a sequential model on static metadata in climate/
   weather ML and is strictly more expressive than additive injection.

3. BIDIRECTIONAL LSTM: The original uni-directional LSTM means that January's
   hidden state knows nothing about what happens in December. A BiLSTM lets
   the SOMSC attention heads for early months still benefit from whole-year
   context, improving monthly predictions without information leakage (since
   we are predicting *within* the same year, not forecasting).

4. RESIDUAL SOMSC (prev_somsc → deltas): The v1 model predicts absolute SOMSC
   values each month. Because SOMSC is a slowly-changing stock variable (often
   ~3000-5000 gC/m²), predicting the small *change* from the previous month's
   prediction makes the learning target much easier and gives a strong
   inductive bias that respects conservation.

5. DEEPER HEADS with LayerNorm + GELU: Single-linear heads limit the
   expressiveness of the pooled representation. Two-layer heads with
   non-linearity and normalization learn richer mappings.

6. DROPOUT everywhere: The original model has zero regularization. We add
   dropout after the daily encoder, in the LSTM, and in the heads.

7. MULTI-HEAD ATTENTION POOLING: The single-head attention pooling is replaced
   with multi-head attention pooling for richer temporal aggregation.

8. SHARED SOMSC TRUNK + MONTH EMBEDDING: Instead of 12 completely independent
   attention+head modules (12×2 = 24 Linear layers), we use a single shared
   attention pooling + a learned month embedding that is concatenated before
   the head. This gives massive parameter savings and lets months share
   statistical strength, while still allowing month-specific predictions.
"""

import torch
from torch import nn
import math

from model.registry import register_model


def month_day_ranges():
    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    starts, ends = [], []
    s = 0
    for m in mdays:
        starts.append(s)
        ends.append(s + m)
        s += m
    return list(zip(starts, ends))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Given a conditioning vector z, learns γ(z) and β(z) to modulate
    an input tensor x as:  output = γ(z) * x + β(z)
    
    This is far more expressive than concatenation for conditioning
    sequential models on static metadata.
    """
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)
        # Initialize gamma to 1 and beta to 0 (identity at init)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.beta_net.bias)
        nn.init.zeros_(self.beta_net.weight)

    def forward(self, x, cond):
        """
        Args:
            x: (B, T, D) sequential features
            cond: (B, C) conditioning vector
        Returns:
            (B, T, D) modulated features
        """
        gamma = self.gamma_net(cond).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_net(cond).unsqueeze(1)     # (B, 1, D)
        return gamma * x + beta


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling over the time dimension.
    
    Uses multiple attention heads to capture different temporal patterns,
    then projects the concatenated result.
    """
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.attn_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_heads)
        ])
        self.proj = nn.Sequential(
            nn.Linear(input_dim * num_heads, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, h, mask=None):
        # h: (B, T, H)
        pooled_heads = []
        attn_weights_all = []
        
        for head in self.attn_heads:
            scores = head(h).squeeze(-1)  # (B, T)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, H)
            pooled_heads.append(pooled)
            attn_weights_all.append(attn_weights)
        
        # Concatenate all heads and project
        combined = torch.cat(pooled_heads, dim=-1)  # (B, H*num_heads)
        output = self.proj(combined)  # (B, output_dim)
        
        # Average attention weights across heads for interpretability
        avg_attn = torch.stack(attn_weights_all, dim=0).mean(dim=0)  # (B, T)
        return output, avg_attn


@register_model("daycent_v2")
class DayCentModelV2(nn.Module):
    """
    Improved DayCent emulator.
    
    Architecture:
        Daily Features ──► DailyEncoder (MLP) ──► FiLM conditioning ──► BiLSTM ──► Attention ──► Heads
                                                       ▲
        Init Cond + Year Enc + Prev SOMSC ──► ContextEncoder ──┘
    
    Outputs:
        - somsc_pred: (B, 12) monthly SOMSC predictions (absolute values via residual chain)
        - yield_pred: (B,) annual yield prediction
    """
    
    def __init__(self, input_dim, init_dim, year_dim,
                 hidden_dim=128, latent_dim=64, lstm_layers=2,
                 dropout=0.1, attn_heads=4, month_emb_dim=16, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ═══════════════════════════════════════════════════
        # 1. DAILY ENCODER — richer than a single linear
        # ═══════════════════════════════════════════════════
        self.daily_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        
        # ═══════════════════════════════════════════════════
        # 2. CONTEXT ENCODER — init cond + year + prev state
        # ═══════════════════════════════════════════════════
        context_input_dim = init_dim + year_dim  # +1 for prev_somsc
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        
        # ═══════════════════════════════════════════════════
        # 3. FiLM CONDITIONING — modulate daily features with context
        # ═══════════════════════════════════════════════════
        self.film = FiLMLayer(cond_dim=latent_dim, feat_dim=latent_dim)
        
        # ═══════════════════════════════════════════════════
        # 4. BIDIRECTIONAL LSTM — captures full-year context
        # ═══════════════════════════════════════════════════
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # BiLSTM outputs 2*hidden_dim; project back
        self.lstm_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # ═══════════════════════════════════════════════════
        # 5. SOMSC HEAD — shared attention + month embedding
        # ═══════════════════════════════════════════════════
        self.somsc_attn = MultiHeadAttentionPooling(
            hidden_dim, hidden_dim, num_heads=attn_heads
        )
        
        # Learned month embeddings so the shared head can specialise
        self.month_embedding = nn.Embedding(12, month_emb_dim)
        
        # Shared SOMSC MLP head (takes pooled repr + month embedding)
        # Predicts a DELTA (change from previous month)
        self.somsc_head = nn.Sequential(
            nn.Linear(hidden_dim + month_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # ═══════════════════════════════════════════════════
        # 6. YIELD HEAD — attention pooled with harvest mask
        # ═══════════════════════════════════════════════════
        self.yield_attn = MultiHeadAttentionPooling(
            hidden_dim, hidden_dim, num_heads=attn_heads
        )
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier init for linear layers, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                # Skip FiLM layers (they have custom init)
                if 'film' not in name:
                    nn.init.xavier_uniform_(param)
    
    def forward(self, batch):
        seq = batch["sequence"]              # (B, 365, F)
        init_cond = batch["init_cond"]       # (B, I)
        year_enc = batch["year_enc"]         # (B, Y)
        harvest_mask = batch["harvest_mask"] # (B, 365)
        prev_somsc = batch["prev_somsc_state"].unsqueeze(-1)  # (B, 1)
        
        B = seq.size(0)
        device = seq.device
        
        # ---- 1. Encode daily features ----
        daily_latent = self.daily_encoder(seq)  # (B, 365, latent)
        
        # ---- 2. Encode static context ----
        context_vec = torch.cat([init_cond, year_enc], dim=-1)
        context_latent = self.context_encoder(context_vec)  # (B, latent)
        
        # ---- 3. FiLM conditioning: modulate daily with context ----
        x = self.film(daily_latent, context_latent)  # (B, 365, latent)
        
        # ---- 4. BiLSTM ----
        h, _ = self.lstm(x)                      # (B, 365, 2*hidden)
        h = self.lstm_proj(h)                     # (B, 365, hidden)
        
        # ---- 5. SOMSC prediction (residual delta chain) ----
        somsc_preds = []
        somsc_attns = []
        ranges = month_day_ranges()
        
        # Start from a learnable transform of prev_somsc
        current_val = prev_somsc.squeeze(-1)  # (B,)
        
        month_ids = torch.arange(12, device=device)  # (12,)
        month_embs = self.month_embedding(month_ids)  # (12, month_emb_dim)
        
        for m, (start, end) in enumerate(ranges):
            # Causal mask: can see days up to end of this month
            mask = torch.zeros(B, 365, dtype=torch.bool, device=device)
            mask[:, :end] = True
            
            # Shared attention pooling
            pooled, attn = self.somsc_attn(h, mask=mask)  # (B, hidden)
            
            # Concatenate month embedding (broadcast across batch)
            m_emb = month_embs[m].unsqueeze(0).expand(B, -1)  # (B, month_emb_dim)
            head_input = torch.cat([pooled, m_emb], dim=-1)    # (B, hidden + month_emb_dim)
            
            # Predict delta and accumulate
            delta = self.somsc_head(head_input).squeeze(-1)    # (B,)
            current_val = current_val + delta
            
            somsc_preds.append(current_val)
            somsc_attns.append(attn)
        
        somsc_preds = torch.stack(somsc_preds, dim=1)  # (B, 12)
        
        # ---- 6. Yield prediction ----
        yield_repr, yield_attn = self.yield_attn(h, mask=harvest_mask)
        yield_pred = self.yield_head(yield_repr).squeeze(-1)  # (B,)
        
        return {
            "somsc_pred": somsc_preds,
            "yield_pred": yield_pred,
            "somsc_attn": somsc_attns,
            "yield_attn": yield_attn,
        }
