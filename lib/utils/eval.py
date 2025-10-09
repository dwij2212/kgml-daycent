import torch

def evaluate(model, loader, device):
    model.eval()
    total_somsc_loss = 0.0
    total_yield_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # move tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)

            # SOMSC masked MSE
            pred_somsc = out["somsc_pred"].squeeze(-1)          # (B, 12)
            target_somsc = batch["somsc"]                      # (B, 12)
            mask_somsc = batch["somsc_mask"]                   # (B, 12) -- float tensor with 1/0 (or cast it)

            mask_sum = mask_somsc.sum()
            if mask_sum.item() > 0:
                somsc_loss = ((pred_somsc - target_somsc)**2 * mask_somsc).sum() / mask_sum
            else:
                somsc_loss = torch.tensor(0.0, device=device)

            # Yield masked MSE
            pred_yield = out["yield_pred"]                     # (B,)
            target_yield = batch["yield"]                      # (B,)
            mask_yield = batch["yield_mask"]                   # (B,)

            mask_y_sum = mask_yield.sum()
            if mask_y_sum.item() > 0:
                yield_loss = ((pred_yield - target_yield)**2 * mask_yield).sum() / mask_y_sum
            else:
                yield_loss = torch.tensor(0.0, device=device)

            bs = batch["sequence"].size(0)
            total_somsc_loss += somsc_loss.item() * bs
            total_yield_loss += yield_loss.item() * bs
            total_samples += bs

    if total_samples == 0:
        return float('nan'), float('nan')
    return total_somsc_loss / total_samples, total_yield_loss / total_samples