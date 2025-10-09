import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class DayCentDataset(Dataset):
    def __init__(self, input_npy_path, output_npy_path, init_cond_path, apply_scaling=True, year_emb_dim=16):
        data_dict = np.load(input_npy_path, allow_pickle=True).item()
        self.data = data_dict["data"]        # (N, 365, #features)
        self.mapping = data_dict["mapping"]  # (N, 3) => (scenario, point_id, year)
        self.columns = list(data_dict["columns"])

        self.init_conditions = self._load_initial_conditions(init_cond_path)

        # Fit scaler only on Tmin/Tmax/Precip
        temp_idx = [i for i, c in enumerate(self.columns) if c in ["Tmin", "Tmax", "Precip"]]
        if apply_scaling:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data[:, :, temp_idx].reshape(-1, len(temp_idx)))
        else:
            self.scaler = None

        self.year_emb_dim = year_emb_dim

        data_dict = np.load(output_npy_path, allow_pickle=True).item()
        self.somsc = data_dict["somsc"]
        self.cgrain = data_dict["cgrain"]

    def _load_initial_conditions(self, path):
        import pandas as pd
        df = pd.read_excel(path).set_index("id")
        df.dropna(axis=1, inplace=True)  # drop columns that are all NaN
        nan_counts = df.isna().sum()
        print(len(nan_counts[nan_counts > 0]))
        #normalise all columns except 'id'
        cols_to_norm = [c for c in df.columns if c != 'id']
        scaler = StandardScaler()
        df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

        return df

    def _year_pos_enc(self, year):
        """Sin-cos positional encoding for year."""
        year_rel = year.astype(int) - 2000
        d = self.year_emb_dim
        pe = np.zeros(d)
        for i in range(0, d, 2):
            div = np.power(10000, 2 * i / d)
            pe[i] = np.sin(year_rel / div)
            pe[i+1] = np.cos(year_rel / div)
        return pe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx].copy()
        sid, year, pid = self.mapping[idx]

         # ---- harvest mask ----
        harvest_idx = np.where(seq[:, self.columns.index("harvest_grain")] == 1)[0]
        if len(harvest_idx) > 0:
            cutoff = harvest_idx[0]  # first harvest day
        else:
            cutoff = 364             # if no harvest, allow whole year
        harvest_mask = np.zeros(365, dtype=np.float32)
        harvest_mask[:cutoff+1] = 1.0

        # --- process doy ---
        doy_idx = self.columns.index("doy")
        doy = seq[:, doy_idx]
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)

        # --- process temps/precip ---
        t_idx = [self.columns.index(c) for c in ["Tmin", "Tmax", "Precip"]]
        if self.scaler is not None:
            seq[:, t_idx] = self.scaler.transform(seq[:, t_idx])

        # --- drop old columns and replace with encoded ---
        keep_idx = [i for i, c in enumerate(self.columns) if c not in ["doy", "Tmin", "Tmax", "Precip"]]
        seq = np.concatenate([
            seq[:, keep_idx], 
            doy_sin[:, None], doy_cos[:, None], 
            seq[:, t_idx]  # standardized Tmin/Tmax/Precip
        ], axis=1)

        # --- initial site conditions ---
        init_cond = self.init_conditions.loc[pid.astype(int)].to_numpy().astype(np.float32)

        # --- year encoding ---
        year_pe = self._year_pos_enc(year).astype(np.float32)

        # ---- labels ----
        somsc = self.somsc[idx]            # shape (12,)
        somsc_mask = ~np.isnan(somsc)      # True = valid
        somsc = np.nan_to_num(somsc, nan=0.0)   # replace NaNs with 0 (ignored by mask)

        yield_val = self.cgrain[idx]   # scalar
        yield_mask = ~np.isnan(yield_val)  # bool
        if np.isnan(yield_val):
            yield_val = 0.0

        return {
        "sequence": torch.tensor(seq, dtype=torch.float32),
        "init_cond": torch.tensor(init_cond, dtype=torch.float32),
        "year_enc": torch.tensor(year_pe, dtype=torch.float32),
        "somsc": torch.tensor(somsc, dtype=torch.float32),
        "somsc_mask": torch.tensor(somsc_mask.astype(np.float32)),
        "yield": torch.tensor(yield_val, dtype=torch.float32),
        "yield_mask": torch.tensor(float(yield_mask)),
        "harvest_mask": torch.tensor(harvest_mask, dtype=torch.float32),
        "pid": pid,
        "year": year
    }