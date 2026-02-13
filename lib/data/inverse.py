"""
DayCent Inverse Modelling Dataset for PyTorch.

Inverse modelling learns latent representations of spatial points from
driver (weather + management) and response (daily output variables)
time-series.

The model learns to:
  1. Reconstruct the input time-series (reconstruction loss)
  2. Predict static site conditions from the latent code (static loss)
  3. Produce similar codes for different years of the same point
     (contrastive loss)

IMPORTANT — normalisation contract
-----------------------------------
The driver_response_df passed to this dataset must ALREADY be normalised.
Use ``prepare_inverse_data()`` (below) to load CSVs, fit scalers on
training points only, and transform the full DataFrame before constructing
dataset instances for any split.
"""
import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional


# ======================================================================
# Data preparation (normalisation)
# ======================================================================
def prepare_inverse_data(
    driver_response_path: str,
    management_path: str,
    init_cond_path: str,
    train_points: List[str],
    train_years: Optional[List[int]] = None,
    scaler_dir: Optional[str] = None,
):
    """
    Load raw CSVs and normalise using statistics from **training points only**.

    Normalises:
        1. Continuous driver-response columns (weather + daily outputs)
           using a StandardScaler fit on rows where point_id ∈ train_points
           (and optionally Year ∈ train_years).
        2. Static site conditions (initial_site_conditions.xlsx) using a
           StandardScaler fit on rows where id ∈ train_points.

    Management columns are left as-is (they are binary 0/1 flags).

    Args:
        driver_response_path: CSV with columns [Year, DOY, point_id, Tmax, …]
        management_path:      CSV with columns [scenario_id, id, Year, doy, …]
        init_cond_path:       Excel with columns [id, …numeric site vars…]
        train_points:         list of point-id strings defining the training set
        train_years:          (optional) restrict scaler fitting to these years too
        scaler_dir:           (optional) directory to save/load fitted scalers

    Returns:
        dict with keys:
            'driver_response_df' : normalised DataFrame  (all points)
            'management_df'      : unchanged DataFrame
            'init_cond_df'       : normalised DataFrame indexed by id (all points)
            'dr_scaler'          : fitted StandardScaler for driver-response
            'static_scaler'      : fitted StandardScaler for static conditions
            'dr_feature_cols'    : list of normalised column names
    """
    train_points_set = set(str(p) for p in train_points)

    # ------------------------------------------------------------------
    # 1. Driver-response
    # ------------------------------------------------------------------
    print("  Loading driver-response CSV …")
    dr_df = pd.read_csv(driver_response_path)
    dr_df['point_id'] = dr_df['point_id'].astype(str)
    dr_df['Year'] = dr_df['Year'].astype(int)
    print(f"    shape: {dr_df.shape}")

    id_cols = {'point_id', 'Year', 'DOY'}
    dr_feature_cols = [c for c in dr_df.columns if c not in id_cols]

    # Fit scaler on training rows only
    train_mask = dr_df['point_id'].isin(train_points_set)
    if train_years is not None:
        train_mask = train_mask & dr_df['Year'].isin(train_years)

    dr_scaler = StandardScaler()
    dr_scaler.fit(dr_df.loc[train_mask, dr_feature_cols])
    n_fit = train_mask.sum()
    print(f"    Fitted driver-response scaler on {n_fit} training rows")
    for i, col in enumerate(dr_feature_cols):
        print(f"      {col:20s}  mean={dr_scaler.mean_[i]:10.4f}  std={dr_scaler.scale_[i]:10.4f}")

    dr_df[dr_feature_cols] = dr_scaler.transform(dr_df[dr_feature_cols])

    # ------------------------------------------------------------------
    # 2. Management  (binary flags — no normalisation needed)
    # ------------------------------------------------------------------
    print("  Loading management CSV …")
    mgmt_df = pd.read_csv(management_path)
    print(f"    shape: {mgmt_df.shape}")

    # ------------------------------------------------------------------
    # 3. Static site conditions
    # ------------------------------------------------------------------
    print("  Loading initial site conditions …")
    init_df = pd.read_excel(init_cond_path).set_index("id")
    init_df.dropna(axis=1, inplace=True)
    static_cols = list(init_df.columns)

    # Fit on training points only
    train_ids_int = []
    for p in train_points:
        try:
            train_ids_int.append(int(p))
        except ValueError:
            train_ids_int.append(p)
    train_init = init_df.loc[init_df.index.isin(train_ids_int)]
    static_scaler = StandardScaler()
    static_scaler.fit(train_init[static_cols])
    print(f"    Fitted static scaler on {len(train_init)} training points")

    init_df[static_cols] = static_scaler.transform(init_df[static_cols])

    print("  Data preparation complete.")
    return {
        'driver_response_df': dr_df,
        'management_df': mgmt_df,
        'init_cond_df': init_df,
        'dr_scaler': dr_scaler,
        'static_scaler': static_scaler,
        'dr_feature_cols': dr_feature_cols,
    }


# ======================================================================
# Dataset
# ======================================================================
class DayCentInverseDataset(Dataset):
    """
    DayCent inverse modelling dataset.

    Each sample is one (scenario, point, year) triple.
    Returns a 365-day sequence of [driver_response + management] channels
    plus the normalized static site conditions as the regression target.

    EXPECTS already-normalised DataFrames (use ``prepare_inverse_data``).
    """

    def __init__(
        self,
        driver_response_df,
        management_df,
        init_cond_df,
        split_config,
        year_emb_dim=16,
    ):
        """
        Args:
            driver_response_df: **Already normalised** DataFrame
            management_df:      Management DataFrame (binary flags, no norm needed)
            init_cond_df:       **Already normalised** DataFrame indexed by point id
            split_config:       dict  {'scenarios': [...], 'points': [...], 'years': [...]}
            year_emb_dim:       dimension for year positional encoding
        """
        print("Initializing DayCent Inverse Dataset ...")

        self.year_emb_dim = year_emb_dim

        # static site conditions (already normalised) -- the regression TARGET
        self.init_conditions = init_cond_df
        self.static_dim = len(init_cond_df.columns)

        # management feature columns (exclude identifiers)
        self.mgmt_cols = [
            c
            for c in management_df.columns
            if c not in ['scenario_id', 'Year', 'doy', 'id']
        ]

        # virtual index mapping
        if not split_config:
            raise ValueError("split_config must be provided")

        self.scenario_ids = [str(s) for s in split_config['scenarios']]
        self.point_ids = [str(p) for p in split_config['points']]
        self.years = [int(y) for y in split_config['years']]

        self.len_scenarios = len(self.scenario_ids)
        self.len_points = len(self.point_ids)
        self.len_years = len(self.years)

        if self.len_scenarios == 0 or self.len_points == 0 or self.len_years == 0:
            self.total_len = 0
            self.year_point_block_size = 0
            self.point_block_size = 0
        else:
            self.total_len = self.len_scenarios * self.len_years * self.len_points
            self.year_point_block_size = self.len_years * self.len_points
            self.point_block_size = self.len_points

        print(
            f"  Virtual index: {self.total_len} samples "
            f"({self.len_scenarios} scenarios x {self.len_years} years "
            f"x {self.len_points} points)"
        )

        # build fast lookup indices
        print("  Building lookup indices ...")
        self._build_lookup_indices(driver_response_df, management_df)
        print(
            f"  Dataset ready  ({self.total_len} samples, "
            f"{self.n_driver_response_feats} driver-response feats, "
            f"{len(self.mgmt_cols)} mgmt feats, "
            f"{self.static_dim} static targets)"
        )

    # ------------------------------------------------------------------
    def _build_lookup_indices(self, driver_response_df, management_df):
        # ===== 1. Driver-Response (weather + daily outputs) =====
        dr = driver_response_df.copy()
        dr['point_id'] = dr['point_id'].astype(str)
        dr['Year'] = dr['Year'].astype(int)

        id_cols = {'point_id', 'Year', 'DOY'}
        self.dr_feature_cols = [c for c in dr.columns if c not in id_cols]
        self.n_driver_response_feats = len(self.dr_feature_cols)

        dr = dr.sort_values(['point_id', 'Year', 'DOY'])
        self.dr_arrays = dr[self.dr_feature_cols].to_numpy(dtype=np.float32)
        self.dr_doy = dr['DOY'].to_numpy(dtype=np.int32)

        gk = dr[['point_id', 'Year']]
        new_group = (gk != gk.shift()).any(axis=1)
        starts = np.where(new_group)[0]
        ends = np.append(starts[1:], len(dr))
        keys = gk.iloc[starts].values

        self.dr_index = {}
        for i in range(len(starts)):
            self.dr_index[(keys[i, 0], int(keys[i, 1]))] = (starts[i], ends[i])

        # ===== 2. Management =====
        mg = management_df.copy()
        mg['scenario_id'] = mg['scenario_id'].astype(str)
        mg['id'] = mg['id'].astype(str)
        mg['Year'] = mg['Year'].astype(int)
        mg = mg.sort_values(['scenario_id', 'id', 'Year', 'doy'])

        self.mgmt_arrays = mg[self.mgmt_cols].to_numpy(dtype=np.float32)
        self.mgmt_doy = mg['doy'].to_numpy(dtype=np.int32)

        gk = mg[['scenario_id', 'id', 'Year']]
        new_group = (gk != gk.shift()).any(axis=1)
        starts = np.where(new_group)[0]
        ends = np.append(starts[1:], len(mg))
        keys = gk.iloc[starts].values

        self.mgmt_index = {}
        for i in range(len(starts)):
            sid, pid, yr = keys[i, 0], keys[i, 1], int(keys[i, 2])
            s, e = starts[i], ends[i]
            doys = self.mgmt_doy[s:e]
            rows = np.arange(s, e)
            self.mgmt_index[(sid, pid, yr)] = list(zip(doys, rows))

        print(
            f"    driver-response index: {len(self.dr_index)} (point, year) groups"
        )
        print(
            f"    management index:      {len(self.mgmt_index)} (scenario, point, year) groups"
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(
                f"Index {idx} out of range [0, {self.total_len})"
            )

        # 1. Decode virtual index
        scenario_idx = idx // self.year_point_block_size
        remainder = idx % self.year_point_block_size
        year_idx = remainder // self.point_block_size
        point_idx = remainder % self.point_block_size

        scenario_id = self.scenario_ids[scenario_idx]
        year = self.years[year_idx]
        pid = self.point_ids[point_idx]

        # 2. Driver-response data (keyed on point, year)
        dr_key = (pid, year)
        n_dr = self.n_driver_response_feats
        n_mg = len(self.mgmt_cols)
        total_feats = n_dr + n_mg

        seq = np.zeros((365, total_feats), dtype=np.float32)

        if dr_key in self.dr_index:
            s, e = self.dr_index[dr_key]
            doys = self.dr_doy[s:e]
            vals = self.dr_arrays[s:e]
            valid = (doys >= 1) & (doys <= 365)
            seq[doys[valid] - 1, :n_dr] = vals[valid]

        # 3. Management data (keyed on scenario, point, year)
        mgmt_key = (scenario_id, pid, year)
        if mgmt_key in self.mgmt_index:
            for doy, row_idx in self.mgmt_index[mgmt_key]:
                if 1 <= doy <= 365:
                    seq[int(doy) - 1, n_dr:] = self.mgmt_arrays[row_idx]

        # 4. Static site conditions -- the regression TARGET
        try:
            init_cond = (
                self.init_conditions.loc[int(pid)].to_numpy().astype(np.float32)
            )
        except KeyError:
            init_cond = np.zeros(self.static_dim, dtype=np.float32)
    
        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "static_target": torch.tensor(init_cond, dtype=torch.float32),
            "pid": pid,
            "year": str(year),
            "scenario_id": scenario_id,
            "point_idx": point_idx,
        }

    # ------------------------------------------------------------------
    @property
    def input_channels(self):
        return self.n_driver_response_feats + len(self.mgmt_cols)

    @property
    def static_channels(self):
        return self.static_dim
