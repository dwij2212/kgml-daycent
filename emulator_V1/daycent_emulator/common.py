"""Shared serialization, feature order, calendar encoding, and metrics."""
from pathlib import Path
import hashlib, json
import numpy as np

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()
EVENT_TYPES = [
    "conventional_till_molboadplow", "corn_planting", "cycle_end", "harvest_grain",
    "herbicide", "notill_rodweederrow", "reduced_till_tandemdisk",
    "ryegrass_planting", "soybean_planting", "winterwheat_planting",
]
INIT_POOL_COLS = ["SOM1CI(2,1)", "SOM2CI(2,1)", "SOM3CI(1)", "SOM2CI(1,1)"]
YEARS = list(range(2000, 2024))
def load_stats(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    # Convert only what is there: a yield run's stats.json predates the pool
    # keys (they arrived with the SOC v2 port), and the rollout loads both.
    for k in ("weather_mean", "weather_std", "static_mean", "static_std",
              "pool_mean", "pool_std", "pool_delta_std",
              "yield_crop_mean", "yield_crop_std"):
        if k in d:
            d[k] = np.array(d[k], dtype=np.float32)
    return d

class GridStore:
    def year_encoding(self, y: int, dim: int = 16) -> np.ndarray:
        rel = int(self.years[y]) - 2000
        pe = np.zeros(dim, dtype=np.float32)
        i = np.arange(0, dim, 2)
        div = np.power(10000.0, 2.0 * i / dim)
        pe[0::2] = np.sin(rel / div)
        pe[1::2] = np.cos(rel / div)
        return pe

def soc_metrics(p,t):
    p=np.asarray(p,np.float64);t=np.asarray(t,np.float64);e=p-t;den=np.square(t-t.mean()).sum()
    return {'n':int(t.size),'r2':float(1-np.square(e).sum()/den) if den>0 else None,
            'rmse_gC_m2':float(np.sqrt(np.square(e).mean())),'bias_gC_m2':float(e.mean())}
