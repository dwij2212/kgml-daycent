import torch

from .training import compute_losses


def evaluate(model, loader, device, somsc_abs_weight=1.0, somsc_delta_weight=10.0):
    """
    Evaluate emulator losses on a data loader.

    Returns averaged losses in normalized space so training and validation use
    the same objective decomposition.
    """
    model.eval()
    total_somsc_abs_loss = 0.0
    total_somsc_delta_loss = 0.0
    total_yield_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch)
            losses = compute_losses(outputs, batch)

            batch_size = batch["sequence"].size(0)
            total_somsc_abs_loss += losses["somsc_abs_loss"].item() * batch_size
            total_somsc_delta_loss += losses["somsc_delta_loss"].item() * batch_size
            total_yield_loss += losses["yield_loss"].item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {
            "somsc_abs_loss": float("nan"),
            "somsc_delta_loss": float("nan"),
            "somsc_loss": float("nan"),
            "yield_loss": float("nan"),
        }

    avg_somsc_abs = total_somsc_abs_loss / total_samples
    avg_somsc_delta = total_somsc_delta_loss / total_samples
    avg_yield = total_yield_loss / total_samples

    return {
        "somsc_abs_loss": avg_somsc_abs,
        "somsc_delta_loss": avg_somsc_delta,
        "somsc_loss": somsc_abs_weight * avg_somsc_abs + somsc_delta_weight * avg_somsc_delta,
        "yield_loss": avg_yield,
    }
