"""Original training-only normalization and yield sampling-bucket recipes."""
import numpy as np
from .common import sha256 as sha
GROUPS = ("earlier_20", "synthetic_expanded_train", "realistic_expanded_train")
def key(sc): return f'{sc["family"]}_{sc["id"]:03d}'


def fit_soc_stats(base, plan):
    """Original soc_improvement_data.fit_normalization formulas and rounding."""
    driver = base / "soc_improvement_v1"
    sites = np.asarray(plan["sites"])
    si = np.searchsorted(sites, plan["train_sites"])
    weather = np.load(driver / "cache/weather.npy", mmap_mode="r")[si].reshape(-1, 3)
    wm, ws = weather.mean(0, dtype=np.float64), weather.std(0, dtype=np.float64)
    del weather
    static = np.load(driver / "cache/static_synthetic.npy", mmap_mode="r")[si]
    sm, ss = static.mean(0, dtype=np.float64), static.std(0, dtype=np.float64)
    pools, deltas, nmax = [], [], 0.0
    for c in plan["scenario_groups"]["earlier_20"]:
        sc = plan["scenarios"][c]
        a = np.load(driver / "targets" / f"{key(sc)}.npy", mmap_mode="r")[:, si]
        pools.append(a.reshape(-1, 4))
        deltas.append(np.diff(a, axis=0).reshape(-1, 4))
        ev = np.load(driver / "cache/management" / sc["family"] / f'{sc["id"]:03d}.npy', mmap_mode="r")
        rates = ev["rate"][(ev["code"] < 0) & np.isin(ev["id"], plan["train_sites"])]
        if len(rates):
            nmax = max(nmax, float(rates.max()))
    pools, delta = np.concatenate(pools), np.concatenate(deltas)
    stats = {
        "weather_mean": wm.astype(np.float32),
        "weather_std": np.where(ws < 1e-6, 1, ws).astype(np.float32),
        "static_mean": sm.astype(np.float32),
        "static_std": np.where(ss < 1e-6, 1, ss).astype(np.float32),
        "nitrogen_scale": max(nmax, 1.0),
        "pool_mean": pools.mean(0, dtype=np.float64).astype(np.float32),
        "pool_std": np.maximum(pools.std(0, dtype=np.float64), 1e-3).astype(np.float32),
        "pool_delta_std": np.maximum(delta.std(0, dtype=np.float64), 1e-3).astype(np.float32),
        "somsc_delta_std": max(float(delta[:, :3].sum(-1, dtype=np.float64).std()), 1e-3),
        "n_soc_transitions": len(delta),
    }
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}

def fit_yield_stats(base, plan, data_dir=None):
    """Same reductions and rounding as soc/yield_improvement_data preparation."""
    driver = base / "soc_improvement_v1"
    sites = np.asarray(plan["sites"])
    ns = len(sites)
    si = np.searchsorted(sites, plan["train_sites"])
    weather = np.load(driver / "cache/weather.npy", mmap_mode="r")[si].reshape(-1, 3)
    wm = weather.mean(0, dtype=np.float64)
    ws = weather.std(0, dtype=np.float64)
    del weather
    static = np.load(driver / "cache/static_synthetic.npy", mmap_mode="r")[si]
    sm = static.mean(0, dtype=np.float64)
    ss = static.std(0, dtype=np.float64)
    pools, deltas, nmax = [], [], 0.0
    for c in plan["scenario_groups"]["earlier_20"]:
        sc = plan["scenarios"][c]
        a = np.load(driver / "targets" / f'{sc["key"]}.npy', mmap_mode="r")[:, si]
        pools.append(a.reshape(-1, 4))
        deltas.append(np.diff(a, axis=0).reshape(-1, 4))
        ev = np.load(driver / "cache/management" / sc["family"] / f'{sc["id"]:03d}.npy', mmap_mode="r")
        rates = ev["rate"][(ev["code"] < 0) & np.isin(ev["id"], plan["train_sites"])]
        if len(rates):
            nmax = max(nmax, float(rates.max()))
    pools, delta = np.concatenate(pools), np.concatenate(deltas)
    stats = {
        "weather_mean": wm.astype(np.float32),
        "weather_std": np.where(ws < 1e-6, 1, ws).astype(np.float32),
        "static_mean": sm.astype(np.float32),
        "static_std": np.where(ss < 1e-6, 1, ss).astype(np.float32),
        "nitrogen_scale": max(nmax, 1.0),
        "pool_mean": pools.mean(0, dtype=np.float64).astype(np.float32),
        "pool_std": np.maximum(pools.std(0, dtype=np.float64), 1e-3).astype(np.float32),
        "pool_delta_std": np.maximum(delta.std(0, dtype=np.float64), 1e-3).astype(np.float32),
        "somsc_delta_std": max(float(delta[:, :3].sum(-1, dtype=np.float64).std()), 1e-3),
        "n_soc_transitions": len(delta),
    }
    del pools, deltas, delta
    target = np.load(base / "yield_improvement_v1/target.npy", mmap_mode="r")
    crop = np.load(base / "yield_improvement_v1/crop.npy", mmap_mode="r")
    assert target.shape == crop.shape == (198, ns, 24)
    tr = np.zeros(ns, bool)
    tr[si] = True
    means, stds, counts, buckets = [], [], [], {}
    for k in range(3):
        vals = []
        for gi, group in enumerate(GROUPS):
            parts = []
            for c in plan["scenario_groups"][group]:
                keep = np.isfinite(target[c]) & (crop[c] == k) & tr[:, None]
                keep[:, 0] = False
                local = np.flatnonzero(keep)
                parts.append((c * ns * 24 + local).astype(np.uint32))
                vals.append(target[c].ravel()[local])
            ix = np.concatenate(parts)
            assert len(ix) and tr[(ix // 24) % ns].all() and (ix % 24 > 0).all()
            assert np.isin(ix // (ns * 24), plan["scenario_groups"][group]).all()
            if data_dir is not None:
                dest = data_dir / "buckets" / f"{gi}_{k}.npy"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest, ix)
                buckets[dest.name] = {"records": len(ix), "sha256": sha(dest)}
        values = np.concatenate(vals)
        means.append(float(values.mean(dtype=np.float64)))
        stds.append(float(values.std(dtype=np.float64)))
        counts.append(len(values))
    stats.update(yield_norm="per_crop", crop_rule="daycent", yield_crop_mean=means,
                 yield_crop_std=stds, yield_crop_count=counts)
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}, buckets
