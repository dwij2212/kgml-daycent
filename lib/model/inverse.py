"""
Inverse model for DayCent: GRU encoder-decoder with attention,
static-attribute prediction head, and contrastive (SimCLR) loss.

Adapted from MODEL_Impute_k.ipynb (hydrology inverse model) to the
DayCent ecosystem modelling context.

Architecture:
    Encoder  : Bidirectional GRU  -> attention-pooled latent code
    Decoder  : Autoregressive GRU -> reconstruction of input sequence
    Static   : Linear head        -> predict static site conditions from code
    Contrastive : SimCLR (NT-Xent) on codes of same-point / different-year pairs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================
# Contrastive loss  (SimCLR / NT-Xent)
# ======================================================================
class SimCLRLoss(nn.Module):
    """
    NT-Xent contrastive loss.

    Expects a tensor z of shape (2*B, D) where the first B rows are
    anchors and the last B rows are positives (same point, different year).
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _mask_correlated(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z):
        """
        Args:
            z: (2*B, D) tensor of latent codes, first B are anchors,
               second B are positives
        Returns:
            scalar loss
        """
        N = z.shape[0]
        batch_size = N // 2

        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / self.temperature

        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).unsqueeze(1)  # (2B, 1)

        mask = self._mask_correlated(batch_size).to(z.device)
        negatives = sim[mask].view(N, -1)  # (2B, 2B-2)

        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        logits = torch.cat([positives, negatives], dim=1)  # (2B, 2B-1)

        loss = self.criterion(logits, labels) / N
        return loss


# ======================================================================
# Encoder-Decoder with Attention
# ======================================================================
class InverseModel(nn.Module):
    """
    GRU-based encoder-decoder for inverse modelling of DayCent.

    Inputs:
        sequence  (B, T, C)    daily time-series (drivers + responses + mgmt)

    Outputs:
        code             (B, code_dim)   latent representation
        reconstruction   (B, T, C)       reconstructed input sequence
        static_pred      (B, S)          predicted static site conditions
    """

    def __init__(
        self,
        in_channels,
        static_channels,
        code_dim=32,
        num_layers=1,
        dropout=0.0,
        device=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self._device = device  # stored for creating zeros in forward

        # Encoder (bidirectional GRU)
        self.encoder = nn.GRU(
            in_channels,
            code_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling over encoder hidden states
        self.att = nn.Linear(code_dim, 1)

        # Decoder (autoregressive GRU)
        self.decoder = nn.GRU(
            in_channels, code_dim, batch_first=True
        )
        self.out_proj = nn.Linear(code_dim, in_channels)

        # Static head:  code -> static site conditions
        self.static_head = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, static_channels),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)

    def encode(self, x):
        """
        Encode a batch of sequences into latent codes.
        Args:
            x: (B, T, C)
        Returns:
            code: (B, code_dim)
        """
        h, _ = self.encoder(x)  # (B, T, 2*code_dim)
        # Merge forward + backward
        h = h[:, :, : self.code_dim] + h[:, :, self.code_dim :]  # (B, T, code_dim)

        # Attention pooling
        att_scores = self.att(h).squeeze(-1)  # (B, T)
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        code = (att_weights * h).sum(dim=1)  # (B, code_dim)
        return code

    def decode(self, code, seq_len, in_channels):
        """
        Autoregressive decoding from the latent code.
        Args:
            code: (B, code_dim)
            seq_len: int (T)
            in_channels: int (C)
        Returns:
            reconstruction: (B, T, C)
        """
        batch = code.shape[0]
        device = code.device

        out = torch.zeros(batch, seq_len, in_channels, device=device)
        h = code.unsqueeze(0)  # (1, B, code_dim) initial hidden state
        inp = torch.zeros(batch, 1, in_channels, device=device)

        for t in range(seq_len):
            dec_out, h = self.decoder(inp, h)  # dec_out: (B, 1, code_dim)
            step_out = self.out_proj(
                dec_out.squeeze(1) * code
            )  # element-wise gating
            out[:, t, :] = step_out
            inp = step_out.unsqueeze(1)

        return out

    def forward(self, batch_dict):
        """
        Full forward pass.

        Args:
            batch_dict: dict with key "sequence" -> (B, T, C)

        Returns:
            dict with keys: code, reconstruction, static_pred
        """
        x = batch_dict["sequence"]  # (B, T, C)
        B, T, C = x.shape

        code = self.encode(x)                        # (B, code_dim)
        reconstruction = self.decode(code, T, C)     # (B, T, C)
        static_pred = self.static_head(code)         # (B, S)

        return {
            "code": code,
            "reconstruction": reconstruction,
            "static_pred": static_pred,
        }
