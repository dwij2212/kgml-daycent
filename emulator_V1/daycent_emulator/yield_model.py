"""Final season-aligned yield LSTM with 730-day driver summaries.

Layer names and initialization order match all three retained checkpoints.
The forward pass reads drivers only; previous yields and SOC are not inputs.
"""
import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return self.proj(pooled), weights


class YieldModel(nn.Module):
    def __init__(self, input_dim=17, init_dim=165, year_dim=16,
                 hidden_dim=256, latent_dim=128, lstm_layers=1):
        super().__init__()
        self.init_proj = nn.Linear(init_dim + year_dim, latent_dim)
        self.daily_proj = nn.Linear(input_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim * 2, hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        self.yield_attn = AttentionPooling(hidden_dim, hidden_dim)
        self.yield_head = nn.Linear(hidden_dim, 1)
        self.antecedent_proj = nn.Linear(64, latent_dim, bias=False)

    def forward(self, batch):
        n = int(batch['season_length'].max())
        sequence = batch['sequence'][:, :n]
        mask = batch['harvest_mask'][:, :n]
        context = self.init_proj(torch.cat([batch['init_cond'], batch['year_enc']], -1))
        context = context + self.antecedent_proj(batch['antecedent_summary'])
        hidden, _ = self.lstm(torch.cat([
            self.daily_proj(sequence), context[:, None].expand(-1, n, -1)], -1))
        pooled, attention = self.yield_attn(hidden, mask)
        return {'yield_pred': self.yield_head(pooled).squeeze(-1),
                'yield_attn': attention}
