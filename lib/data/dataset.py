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
    DayCent dataset with on-the-fly data merging and flexible splits.
    
    This dataset performs data merging during iteration, allowing for:
    - Flexible train/val/test splits without data duplication
    - Support for overlapping scenarios, points, and years across splits
    
    Usage:
        Create separate instances for train/val/test:
        train_dataset = DayCentDatasetV2(..., split_config=train_config)
        val_dataset = DayCentDatasetV2(..., split_config=val_config)
    """
    
    def __init__(self, weather_df, management_df, output_df, init_cond_path,
                 split_config, year_emb_dim=16):
        """
        Args:
            weather_df: DataFrame with weather data (already normalized)
            management_df: DataFrame with management events (has 'scenario_id' column with strings)
            output_df: DataFrame with outputs (already normalized, has 'scenario_id' column with strings)
            init_cond_path: Path to initial conditions Excel file
            split_config: Dict with keys 'scenarios', 'points', 'years' defining this split
            year_emb_dim: Dimension for year positional encoding
        """
        print("Initializing optimized dynamic dataset...")
        
        self.year_emb_dim = year_emb_dim
        self.init_conditions = self._load_initial_conditions(init_cond_path)
        
        # Get management feature columns (exclude identifiers)
        self.mgmt_cols = [c for c in management_df.columns 
                         if c not in ['scenario_id', 'Year', 'doy']]
        
        # Create VIRTUAL index mapping for this split
        if not split_config:
            raise ValueError("split_config must be provided")
        
        # Store the lists of keys, not the materialized cartesian product
        self.scenario_ids = [str(s) for s in split_config['scenarios']]
        self.point_ids = [str(p) for p in split_config['points']]
        self.years = split_config['years'] # Assumed to be list of ints
        
        self.len_scenarios = len(self.scenario_ids)
        self.len_points = len(self.point_ids)
        self.len_years = len(self.years)
        
        if self.len_scenarios == 0 or self.len_points == 0 or self.len_years == 0:
            self.total_len = 0
            self.year_point_block_size = 0
            self.point_block_size = 0
        else:
            # This is the order from your original _create_index_mapping:
            # for scenario... for year... for point...
            self.total_len = self.len_scenarios * self.len_years * self.len_points
            self.year_point_block_size = self.len_years * self.len_points
            self.point_block_size = self.len_points
        
        print(f"  Created virtual index mapping with {self.total_len} samples")
        
        # Pre-build lookup indices for O(1) access
        print("  Building fast lookup indices...")
        self._build_lookup_indices_optimized(weather_df, management_df, output_df)
        
        print(f"Dataset initialized with {self.total_len} samples")
    
    
    def _build_lookup_indices_optimized(self, weather_df, management_df, output_df):
        """
        Pre-build indices for fast O(1) lookups using sorting and group boundary detection.
        This is significantly faster than the original loop-and-mask approach.
        """
        
        # --- 1. Process Weather Data ---
        print("    Processing weather data...")
        # Ensure correct types and sort
        weather_df['point_id'] = weather_df['point_id'].astype(str)
        weather_df['Year'] = weather_df['Year'].astype(int)
        weather_df = weather_df.sort_values(['point_id', 'Year', 'doy'])
        
        self.weather_arrays = weather_df[['Tmax', 'Tmin', 'Precip']].to_numpy()
        self.weather_doy = weather_df['doy'].to_numpy()
        
        # Find group boundaries
        group_keys_df = weather_df[['point_id', 'Year']]
        # .ne() is "not equal", .shift() moves data down, .any(axis=1) checks row-wise
        is_new_group = (group_keys_df != group_keys_df.shift()).any(axis=1)
        group_start_indices = np.where(is_new_group)[0]
        group_end_indices = np.append(group_start_indices[1:], len(weather_df))
        
        # Get the keys for each group (at the start index)
        group_keys = group_keys_df.iloc[group_start_indices].values
        
        # Build hash map in one pass
        self.weather_index = {}
        for i in range(len(group_start_indices)):
            # key_tuple is (pid_str, year_int)
            key_tuple = (group_keys[i, 0], group_keys[i, 1]) 
            self.weather_index[key_tuple] = (group_start_indices[i], group_end_indices[i])
        
        
        # --- 2. Process Management Data ---
        print("    Processing management data...")
        # Ensure correct types and sort
        management_df['scenario_id'] = management_df['scenario_id'].astype(str)
        management_df['Year'] = management_df['Year'].astype(int)
        management_df = management_df.sort_values(['scenario_id', 'Year', 'doy'])
        
        self.mgmt_arrays = management_df[self.mgmt_cols].to_numpy()
        self.mgmt_doy = management_df['doy'].to_numpy()
        
        # Find group boundaries
        group_keys_df = management_df[['scenario_id', 'Year']]
        is_new_group = (group_keys_df != group_keys_df.shift()).any(axis=1)
        group_start_indices = np.where(is_new_group)[0]
        group_end_indices = np.append(group_start_indices[1:], len(management_df))
        group_keys = group_keys_df.iloc[group_start_indices].values
        
        # Build hash map
        self.mgmt_index = {}
        for i in range(len(group_start_indices)):
            key_tuple = (group_keys[i, 0], group_keys[i, 1]) # (sid_str, year_int)
            start, end = group_start_indices[i], group_end_indices[i]
            # Store as list of (doy, row_idx) for this scenario-year
            # We use the original indices from the sorted array
            doys_in_group = self.mgmt_doy[start:end]
            indices_in_group = np.arange(start, end)
            self.mgmt_index[key_tuple] = list(zip(doys_in_group, indices_in_group))
            
            
        # --- 3. Process Output Data ---
        print("    Processing output data...")
        # Ensure correct types and sort
        output_df['scenario_id'] = output_df['scenario_id'].astype(str)
        output_df['point_id'] = output_df['point_id'].astype(str)
        output_df['Year'] = output_df['Year'].astype(int)
        output_df = output_df.sort_values(['scenario_id', 'point_id', 'Year', 'month'])
        
        self.output_somsc = output_df['somsc'].to_numpy()
        self.output_cgrain = output_df['cgrain'].to_numpy()
        self.output_month = output_df['month'].to_numpy()
        
        # Find group boundaries
        group_keys_df = output_df[['scenario_id', 'point_id', 'Year']]
        is_new_group = (group_keys_df != group_keys_df.shift()).any(axis=1)
        group_start_indices = np.where(is_new_group)[0]
        group_end_indices = np.append(group_start_indices[1:], len(output_df))
        group_keys = group_keys_df.iloc[group_start_indices].values
        
        # Build hash map
        self.output_index = {}
        for i in range(len(group_start_indices)):
            key_tuple = (group_keys[i, 0], group_keys[i, 1], group_keys[i, 2]) # (sid_str, pid_str, year_int)
            start, end = group_start_indices[i], group_end_indices[i]
            # Store the range of indices (which are contiguous)
            self.output_index[key_tuple] = np.arange(start, end)
            
        print("    Lookup indices built successfully!")
    
    def _load_initial_conditions(self, path):
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
            pe[i+1] = np.cos(year_rel / div)
        return pe

    def __len__(self):
        # Return the pre-calculated total length
        return self.total_len

    def __getitem__(self, idx):
        if idx >= self.total_len or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.total_len}")

        # 1. Calculate (scenario, year, pid) from idx (VIRTUAL INDEX)
        # This replaces: scenario_id, year, pid = self.index_mapping[idx]
        
        scenario_idx = idx // self.year_point_block_size
        remainder = idx % self.year_point_block_size
        year_idx = remainder // self.point_block_size
        point_idx = remainder % self.point_block_size
        
        scenario_id = self.scenario_ids[scenario_idx] # str
        year = self.years[year_idx]                 # int
        pid = self.point_ids[point_idx]               # str

        prev_year = year - 1
        
        # 2. Fetch Previous Year's Output (The State)
        # We need the LAST valid value from the previous year (usually Month 12)
        prev_key = (scenario_id, pid, prev_year)
        prev_somsc_state = 0.0 # Default if t=0 or missing
        
        if prev_key in self.output_index:
            indices = self.output_index[prev_key]
            last_idx = indices[-1] 
            val = self.output_somsc[last_idx]
            
            if not np.isnan(val):
                prev_somsc_state = val
        
        
        # 2. Get weather data using pre-computed index (O(1) hash lookup)
        weather_key = (pid, year)
        if weather_key in self.weather_index:
            start_idx, end_idx = self.weather_index[weather_key]
            weather_vals = self.weather_arrays[start_idx:end_idx]
            weather_doys = self.weather_doy[start_idx:end_idx]
        else:
            # No weather data for this point-year
            weather_vals = np.zeros((0, 3), dtype=np.float32)
            weather_doys = np.array([], dtype=np.int32)
        
        # 3. Get management data using pre-computed index (O(1) hash lookup)
        mgmt_key = (scenario_id, year)
        if mgmt_key in self.mgmt_index:
            mgmt_events = self.mgmt_index[mgmt_key]  # List of (doy, row_idx)
        else:
            mgmt_events = []
        
        # 4. Create daily sequence (365 days) efficiently
        seq = np.zeros((365, 3 + len(self.mgmt_cols)), dtype=np.float32)
        
        # Fill in weather data (map doy to array index)
        for i, doy in enumerate(weather_doys):
            if 1 <= doy <= 365:
                seq[int(doy) - 1, :3] = weather_vals[i]
        
        # Fill in management data (map doy to array index)
        for doy, row_idx in mgmt_events:
            if 1 <= doy <= 365:
                seq[int(doy) - 1, 3:] = self.mgmt_arrays[row_idx]
        
        # 5. Add doy column at the beginning
        doy_col = np.arange(1, 366, dtype=np.float32).reshape(-1, 1)
        seq = np.concatenate([doy_col, seq], axis=1)
        
        # 6. Process doy (sin/cos encoding)
        doy = seq[:, 0]
        doy_sin = np.sin(2 * np.pi * doy / 365).reshape(-1, 1)
        doy_cos = np.cos(2 * np.pi * doy / 365).reshape(-1, 1)
        seq = np.concatenate([seq, doy_sin, doy_cos], axis=1)
        
        # 7. Harvest mask - find 'harvest_grain' column if it exists
        # The columns are: [doy, Tmax, Tmin, Precip, ...mgmt_cols..., doy_sin, doy_cos]
        if 'harvest_grain' in self.mgmt_cols:
            harvest_col_idx = 4 + self.mgmt_cols.index('harvest_grain')  # 4 = doy + 3 weather cols
            harvest_idx = np.where(seq[:, harvest_col_idx] == 1)[0]
            cutoff = harvest_idx[0] if len(harvest_idx) > 0 else 364
        else:
            cutoff = 364
        harvest_mask = np.zeros(365, dtype=np.float32)
        harvest_mask[:cutoff+1] = 1.0
        
        # 8. Initial site conditions
        init_cond = self.init_conditions.loc[int(pid)].to_numpy().astype(np.float32)
        
        # 9. Year encoding
        year_pe = self._year_pos_enc(year).astype(np.float32)
        
        # 10. Get outputs using pre-computed index (O(1) hash lookup)
        output_key = (scenario_id, pid, year)
        somsc_array = np.full(12, np.nan, dtype=np.float32)
        yield_val = 0.0
        yield_mask = 0.0
        
        if output_key in self.output_index:
            indices = self.output_index[output_key] # This is now np.arange(start, end)
            
            # Process SOMSC (monthly values)
            for idx_pos in indices:
                month = int(self.output_month[idx_pos])
                somsc_val = self.output_somsc[idx_pos]
                if not np.isnan(somsc_val) and 1 <= month <= 12:
                    somsc_array[month - 1] = somsc_val
            
            # Process CGRAIN (annual value) - take first non-NaN value
            cgrain_vals = self.output_cgrain[indices]
            valid_cgrain = cgrain_vals[~np.isnan(cgrain_vals)]
            if len(valid_cgrain) > 0:
                yield_val = valid_cgrain[0]
                yield_mask = 1.0
        
        somsc_mask = ~np.isnan(somsc_array)
        somsc_array = np.nan_to_num(somsc_array, nan=0.0)
        
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
            "year": str(year), # return str for consistency
            "scenario_id": scenario_id,
            "prev_somsc_state": torch.tensor(prev_somsc_state, dtype=torch.float32),
        }