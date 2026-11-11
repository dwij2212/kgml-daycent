"""Final daily LSTM and annual four-pool SOC transition.

The annual state is updated additively. The first three pools sum to SOMSC;
surface slow carbon is the fourth carried pool. Eleven monthly outputs are
auxiliary training supervision. No positivity correction is applied.
"""
import torch
from torch import nn

SOC_STATE_COLS = ['som1c_soil', 'som2c_soil', 'som3c', 'som2c_surface']


class SOCModel(nn.Module):
    def __init__(self, arm='s_monthly_width512', input_dim=16, init_dim=165,
                 year_dim=16, hidden_dim=512, latent_dim=128, lstm_layers=1,
                 dropout=0.1, unroll=8, monthly_loss_weight=0.5,
                 somsc_loss_weight=0.5):
        super().__init__()
        self.experiment_arm = arm
        self.training_epoch = 1
        self.unroll = unroll
        self.monthly_loss_weight = monthly_loss_weight
        self.somsc_loss_weight = somsc_loss_weight
        # Preserve the original layer construction order and checkpoint keys.
        self.monthly_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.context_proj = nn.Sequential(
            nn.Linear(init_dim + year_dim + 4, latent_dim), nn.LayerNorm(latent_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(latent_dim * 2, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))
        self.monthly_head = nn.Linear(hidden_dim, 11 * 4)

    def configure_epoch(self, epoch):
        self.training_epoch = epoch
        return self.unroll

    def forward(self, batch):
        sequence = batch['sequence']
        latent = self.monthly_proj(sequence)
        context = self.context_proj(torch.cat([
            batch['init_cond'], batch['year_enc'],
            batch['prev_soc_state_norm'][:, [0, 1, 2, 3]]], -1))
        hidden, _ = self.lstm(torch.cat([
            latent, context[:, None].expand(-1, sequence.shape[1], -1)], -1))
        features = self.output_head[:-1](hidden[:, -1])
        delta = self.output_head[-1](features)
        state = batch['prev_soc_state_raw'] + delta
        monthly = batch['prev_soc_state_raw'][:, None] + self.monthly_head(features).reshape(-1, 11, 4)
        return {'soc_state_pred': state, 'somsc_pred': state[:, :3].sum(-1),
                'soc_state_delta_pred': delta, 'monthly_state_pred': monthly}


def year_batch(batch, state, y, stats, arm='monthly_k8'):
    mean = torch.as_tensor(stats['pool_mean'], device=state.device)
    std = torch.as_tensor(stats['pool_std'], device=state.device)
    return {'sequence': batch['sequence'][:, y], 'year_enc': batch['year_enc'][:, y],
            'init_cond': batch['init_cond'], 'prev_soc_state_raw': state,
            'prev_soc_state_norm': (state - mean) / std,
            'prev_somsc_state': state[:, :3].sum(-1)}


def state_error(pred, true, stats, somsc_weight=0.5):
    std = torch.as_tensor(stats['pool_delta_std'], device=pred.device)
    pools = ((pred - true) / std).square().mean(-1)
    total = ((pred[..., :3].sum(-1) - true[..., :3].sum(-1)) /
             float(stats['somsc_delta_std'])).square()
    return pools + somsc_weight * total
