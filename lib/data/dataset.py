"""
DayCent Dataset implementations for PyTorch.

This module provides two dataset classes:
1. DayCentDataset: Legacy dataset that loads pre-processed .npy files
2. DayCentDatasetV2: Modern dataset with on-the-fly merging and flexible splits
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class DayCentDataset(Dataset):
    """
    Legacy dataset for loading pre-processed DayCent data from .npy files.
    
    This dataset expects data to be pre-processed and saved as numpy arrays.
    Use this for backward compatibility with existing experiments.
    """
    
    def __init__(self, input_npy_path, output_npy_path, init_cond_path, year_emb_dim=16):
        data_dict = np.load(input_npy_path, allow_pickle=True).item()
        self.data = data_dict["data"]        # (N, 365, #features)
        self.mapping = data_dict["mapping"]  # (N, 3) => (scenario, point_id, year)
        self.columns = list(data_dict["columns"])

        self.init_conditions = self._load_initial_conditions(init_cond_path)
        self.year_emb_dim = year_emb_dim

        data_dict = np.load(output_npy_path, allow_pickle=True).item()
        self.somsc = data_dict["somsc"]
        self.cgrain = data_dict["cgrain"]

    def _load_initial_conditions(self, path):
        df = pd.read_excel(path).set_index("id")
        df.dropna(axis=1, inplace=True)  # drop columns that are all NaN

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

        seq = np.concatenate([seq, doy_sin[:, None], doy_cos[:, None]], axis=1)

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


class DayCentDatasetV2(Dataset):
    """
    DayCent dataset with on-the-fly data merging and flexible train/val/test splits.
    
    This dataset performs data merging during iteration, allowing for:
    - Flexible train/val/test splits without data duplication
    - Runtime mode switching (train -> val -> test)
    - Support for overlapping scenarios, points, and years across splits
    
    """
    
    def __init__(self, weather_df, management_df, output_df, init_cond_path,
                 train_config=None, val_config=None, test_config=None,
                 dataset_type='train', year_emb_dim=16):
        """
        Initialize DayCentDatasetV2.
        
        Args:
            weather_df: DataFrame with weather data (already normalized)
            management_df: DataFrame with management events
            output_df: DataFrame with outputs (already normalized)
            init_cond_path: Path to initial conditions Excel file
            train_config: Dict with keys 'scenarios', 'points', 'years'
            val_config: Dict with keys 'scenarios', 'points', 'years'
            test_config: Dict with keys 'scenarios', 'points', 'years'
            dataset_type: One of 'train', 'val', 'test'
            year_emb_dim: Dimension for year positional encoding
        """
        self.weather_df = weather_df
        self.management_df = management_df
        self.output_df = output_df
        self.init_conditions = self._load_initial_conditions(init_cond_path)
        self.year_emb_dim = year_emb_dim
        self.dataset_type = dataset_type
        
        self.mgmt_cols = [c for c in self.management_df.columns 
                         if c not in ['scenario_id', 'Year', 'doy']]
        
        self.train_mapping = self._create_index_mapping(train_config) if train_config else []
        self.val_mapping = self._create_index_mapping(val_config) if val_config else []
        self.test_mapping = self._create_index_mapping(test_config) if test_config else []
        
        self._set_dataset_type(dataset_type)
        print(f"Dataset initialized in '{dataset_type}' mode with {len(self)} samples")
    
    def _create_index_mapping(self, config):
        """Create index mapping for given scenarios, points, and years."""
        if not config:
            return []
        
        scenario_ids = [str(s) for s in config['scenarios']]
        point_ids = [str(p) for p in config['points']]
        years = config['years']
        
        mapping = []
        for scenario_id in scenario_ids:
            for year in years:
                for point_id in point_ids:
                    mapping.append((scenario_id, year, point_id))
        return mapping
    
    def _set_dataset_type(self, dataset_type):
        """Set the active dataset type and mapping."""
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError(f"dataset_type must be 'train', 'val', or 'test', got '{dataset_type}'")
        
        self.dataset_type = dataset_type
        
        if dataset_type == 'train':
            self.index_mapping = self.train_mapping
        elif dataset_type == 'val':
            self.index_mapping = self.val_mapping
        else:
            self.index_mapping = self.test_mapping
        
        if len(self.index_mapping) == 0:
            print(f"Warning: No samples for {dataset_type} split")
    
    def set_mode(self, dataset_type):
        """
        Switch between train/val/test modes at runtime.
        
        Args:
            dataset_type: One of 'train', 'val', 'test'
        """
        self._set_dataset_type(dataset_type)
        print(f"Dataset mode changed to '{dataset_type}' with {len(self)} samples")

    def _load_initial_conditions(self, path):
        """Load and normalize initial site conditions."""
        df = pd.read_excel(path).set_index("id")
        df.dropna(axis=1, inplace=True)
        cols_to_norm = [c for c in df.columns if c != 'id']
        scaler = StandardScaler()
        df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
        return df

    def _year_pos_enc(self, year):
        """Sin-cos positional encoding for year."""
        year_rel = int(year) - 2000
        d = self.year_emb_dim
        pe = np.zeros(d)
        for i in range(0, d, 2):
            div = np.power(10000, 2 * i / d)
            pe[i] = np.sin(year_rel / div)
            if i + 1 < d:
                pe[i+1] = np.cos(year_rel / div)
        return pe

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        scenario_id, year, pid = self.index_mapping[idx]
        
        weather_data = self.weather_df[
            (self.weather_df['point_id'] == pid) & 
            (self.weather_df['Year'] == year)
        ].sort_values('doy')
        
        mgmt_data = self.management_df[
            (self.management_df['scenario_id'] == scenario_id) & 
            (self.management_df['Year'] == year)
        ].sort_values('doy')
        
        daily_df = pd.DataFrame({'doy': range(1, 366)})
        daily_df = daily_df.merge(weather_data[['doy', 'Tmax', 'Tmin', 'Precip']], 
                                  on='doy', how='left')
        daily_df = daily_df.merge(mgmt_data[['doy'] + self.mgmt_cols], 
                                  on='doy', how='left')
        daily_df.fillna(0, inplace=True)
        
        feature_cols = ['doy', 'Tmax', 'Tmin', 'Precip'] + self.mgmt_cols
        seq = daily_df[feature_cols].to_numpy().astype(np.float32)
        
        doy = seq[:, 0]
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        seq = np.concatenate([seq, doy_sin[:, None], doy_cos[:, None]], axis=1)
        
        harvest_col_idx = feature_cols.index('harvest_grain') if 'harvest_grain' in feature_cols else -1
        if harvest_col_idx >= 0:
            harvest_idx = np.where(seq[:, harvest_col_idx] == 1)[0]
            cutoff = harvest_idx[0] if len(harvest_idx) > 0 else 364
        else:
            cutoff = 364
        harvest_mask = np.zeros(365, dtype=np.float32)
        harvest_mask[:cutoff+1] = 1.0
        
        init_cond = self.init_conditions.loc[int(pid)].to_numpy().astype(np.float32)
        year_pe = self._year_pos_enc(year).astype(np.float32)
        
        output_data = self.output_df[
            (self.output_df['scenario_id'] == scenario_id) &
            (self.output_df['point_id'] == pid) &
            (self.output_df['Year'] == year)
        ]
        
        somsc_array = np.full(12, np.nan, dtype=np.float32)
        if not output_data.empty:
            for _, row in output_data.iterrows():
                month = int(row['month'])
                if 1 <= month <= 12 and not pd.isna(row['somsc']):
                    somsc_array[month - 1] = row['somsc']
        
        somsc_mask = ~np.isnan(somsc_array)
        somsc_array = np.nan_to_num(somsc_array, nan=0.0)
        
        cgrain_values = output_data['cgrain'].dropna()
        if len(cgrain_values) > 0:
            yield_val = cgrain_values.iloc[0]
            yield_mask = 1.0
        else:
            yield_val = 0.0
            yield_mask = 0.0
        
        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "init_cond": torch.tensor(init_cond, dtype=torch.float32),
            "year_enc": torch.tensor(year_pe, dtype=torch.float32),
            "somsc": torch.tensor(somsc_array, dtype=torch.float32),
            "somsc_mask": torch.tensor(somsc_mask.astype(np.float32)),
            "yield": torch.tensor(yield_val, dtype=torch.float32),
            "yield_mask": torch.tensor(yield_mask, dtype=torch.float32),
            "harvest_mask": torch.tensor(harvest_mask, dtype=torch.float32),
            "pid": pid,
            "year": str(year),
            "scenario_id": scenario_id
        }