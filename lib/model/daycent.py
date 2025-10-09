import torch
from torch import nn

def month_day_ranges():
    mdays = [31,28,31,30,31,30,31,31,30,31,30,31]
    starts, ends = [], []
    s = 0
    for m in mdays:
        starts.append(s)
        ends.append(s + m)   # exclusive
        s += m
    return list(zip(starts, ends))   # 0-indexed day ranges

class AttentionPooling(nn.Module):
    """Generic attention pooling over time dimension."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, h, mask=None):
        # h: (B, T, H)
        # mask: (B, T) binary mask (1=keep, 0=ignore)
        scores = self.attn(h).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return self.proj(pooled), attn_weights


class DayCentModel(nn.Module):
    def __init__(self, input_dim, init_dim, year_dim, hidden_dim=128, latent_dim=32, lstm_layers=2):
        super().__init__()

        # project init conditions + year encoding → latent feature
        self.init_proj = nn.Linear(init_dim + year_dim, latent_dim)

        # project daily inputs → latent feature
        self.daily_proj = nn.Linear(input_dim, latent_dim)

        # combine both
        self.lstm = nn.LSTM(
            input_size=latent_dim * 2, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True
        )

        # Attention heads
        self.somsc_attn = AttentionPooling(hidden_dim, hidden_dim)
        self.yield_attn = AttentionPooling(hidden_dim, hidden_dim)

        # Output layers
        self.somsc_head = nn.Linear(hidden_dim, 1)  # monthly but gets stacked in forward pass
        self.yield_head = nn.Linear(hidden_dim, 1)   # yearly

    def forward(self, batch):
        seq = batch["sequence"]             # (B, 365, F)
        init_cond = batch["init_cond"]      # (B, I)
        year_enc = batch["year_enc"]        # (B, Y)
        harvest_mask = batch["harvest_mask"]# (B, 365)

        # ---- daily representation ----
        daily_latent = self.daily_proj(seq)  # (B, 365, latent)
        
        # ---- global representation (init+year) ----
        global_latent = self.init_proj(torch.cat([init_cond, year_enc], dim=-1))  # (B, latent)

        global_latent = global_latent.unsqueeze(1).repeat(1, seq.size(1), 1)      # (B, 365, latent)

        # ---- concat + LSTM ----
        x = torch.cat([daily_latent, global_latent], dim=-1)  # (B, 365, 2*latent)
        h, _ = self.lstm(x)                                   # (B, 365, H)

        # ---- SOMSC head ----
        somsc_preds = []
        somsc_attns = []
        ranges = month_day_ranges()
        for m, (start, end) in enumerate(ranges):
            mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
            mask[:, 0:end] = True
            pooled, attn = self.somsc_attn(h, mask=mask)
            pred = self.somsc_head(pooled)
            somsc_preds.append(pred)
            somsc_attns.append(attn)
        somsc_preds = torch.stack(somsc_preds, dim=1)  # (B,12)

        # ---- Yield head ----
        yield_repr, yield_attn = self.yield_attn(h, mask=harvest_mask)  # (B, H)
        yield_pred = self.yield_head(yield_repr).squeeze(-1)            # (B,)

        return {
            "somsc_pred": somsc_preds,
            "yield_pred": yield_pred,
            "somsc_attn": somsc_attns,
            "yield_attn": yield_attn
        }