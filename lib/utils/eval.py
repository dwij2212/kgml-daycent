import torch

from .training import compute_masked_mse

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
            

            # SOMSC loss
            if "somsc_delta_pred" in out and "somsc_deltas" in batch:
                
                # Primary Objective: Match the Rate of Change (Deltas)
                somsc_loss = compute_masked_mse(
                    out["somsc_delta_pred"], 
                    batch["somsc_deltas"], 
                    batch["somsc_delta_mask"]
                )
                
            else:
                # Fallback for legacy models (predicting absolute only)
                somsc_loss = compute_masked_mse(
                    out["somsc_pred"], 
                    batch["somsc"], 
                    batch["somsc_mask"]
                )

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