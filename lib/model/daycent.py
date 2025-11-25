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

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # We learn log_vars (log variance) for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_somsc, loss_yield):
        # Task 1: SOMSC
        precision1 = torch.exp(-self.log_vars[0])
        weighted_loss1 = precision1 * loss_somsc + self.log_vars[0]

        # Task 2: Yield
        precision2 = torch.exp(-self.log_vars[1])
        weighted_loss2 = precision2 * loss_yield + self.log_vars[1]

        return weighted_loss1 + weighted_loss2
    
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

        # for previous somsc state
        self.state_proj = nn.Linear(1, latent_dim)

        # project init conditions + year encoding → latent feature
        self.init_proj = nn.Linear(init_dim + year_dim + latent_dim, latent_dim)

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
        # FIX 1: Separate attention for each month
        self.somsc_attns = nn.ModuleList([
            AttentionPooling(hidden_dim, hidden_dim) for _ in range(12)
        ])
        
        # FIX 2: Separate output head for each month
        self.somsc_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(12)
        ])
        
        # Yield prediction
        self.yield_attn = AttentionPooling(hidden_dim, hidden_dim)
        self.yield_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        seq = batch["sequence"]             # (B, 365, F)
        init_cond = batch["init_cond"]      # (B, I)
        # make init cond 0 for ablation
        # init_cond = torch.zeros_like(init_cond)
        year_enc = batch["year_enc"]        # (B, Y)
        harvest_mask = batch["harvest_mask"]# (B, 365)
        prev_somsc = batch["prev_somsc_state"].unsqueeze(-1)  # (B, 1)

        state_latent = self.state_proj(prev_somsc)  # (B, latent)

        # ---- daily representation ----
        daily_latent = self.daily_proj(seq)  # (B, 365, latent)
        
        # ---- global representation (init+year) ----
        global_latent = self.init_proj(torch.cat([init_cond, year_enc, state_latent], dim=-1))  # (B, latent)

        global_latent = global_latent.unsqueeze(1).repeat(1, seq.size(1), 1)      # (B, 365, latent)

        # ---- concat + LSTM ----
        x = torch.cat([daily_latent, global_latent], dim=-1)  # (B, 365, 2*latent)
        h, _ = self.lstm(x)                                   # (B, 365, H)

        # ---- SOMSC head (per month) ----
        somsc_preds = []
        somsc_attns = []
        ranges = month_day_ranges()

        current_val = prev_somsc.squeeze(-1) # (B, )


        for m, (start, end) in enumerate(ranges):
            # Create mask: can see days [0, end)
            mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
            mask[:, :end] = True  # Clearer syntax
            
            # Use month-specific attention and head
            pooled, attn = self.somsc_attns[m](h, mask=mask)
            delta = self.somsc_heads[m](pooled).squeeze(-1) # (B, )
            
            # The absolute prediction is accumulation of deltas
            current_val = current_val + delta
            somsc_preds.append(current_val)
            somsc_attns.append(attn)
        
        somsc_preds = torch.stack(somsc_preds, dim=1).squeeze(-1)  # (B, 12)
        
        # ---- Yield head ----
        yield_repr, yield_attn = self.yield_attn(h, mask=harvest_mask)
        yield_pred = self.yield_head(yield_repr).squeeze(-1)            # (B,)

        return {
            "somsc_pred": somsc_preds,
            "yield_pred": yield_pred,
            "somsc_attn": somsc_attns,
            "yield_attn": yield_attn
        }