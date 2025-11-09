"""
Standalone test file for experimenting with dataset loader.

This file consolidates all relevant preprocessing and dataset loading logic
so you can experiment with moving merging logic into __getitem__.

Usage:
    python test_dataset_loader.py
"""
import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from glob import glob
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS (Unchanged)
# ============================================================================

def load_weather_data(weather_dir: str) -> pd.DataFrame:
    """Load and concatenate all weather data files."""
    all_points = []
    for points in os.listdir(weather_dir):
        if not points.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(weather_dir, points))
        df['point_id'] = points.split(".csv")[0]
        all_points.append(df)
    weather_df = pd.concat(all_points, ignore_index=True)
    return weather_df


def load_single_scenario_output(scenario_id: str, output_dir: str):
    """Load output data for a single scenario."""
    month_to_doy = {1:30, 2:58, 3:89, 4:119, 5:150, 6:180, 7:211, 8:242, 9:272, 10:303, 11:333, 12:364}
    
    monthly_df = pd.read_csv(os.path.join(output_dir, f"SAS_scenario_{scenario_id}_monthly.csv"))
    monthly_df = monthly_df.rename({'id': 'point_id'}, axis=1)
    monthly_df['doy'] = monthly_df['month'].map(month_to_doy)
    monthly_df['simyear'] = monthly_df['simyear'].apply(lambda x: math.floor(float(x)))

    harvest_df = pd.read_csv(os.path.join(output_dir, f"SAS_scenario_{scenario_id}_harvest.csv"))
    harvest_df = harvest_df.rename({'id': 'point_id', 'dayofyr': 'doy'}, axis=1)

    output_df = pd.merge(monthly_df, harvest_df, on=['runid', 'point_id', 'simyear', 'doy'], how='outer')
    output_df = output_df.rename({'simyear': 'Year'}, axis=1)
    output_df['point_id'] = output_df['point_id'].astype(str)

    dates = pd.to_datetime(output_df['Year'].astype(str) + '-' + output_df['doy'].astype(str), format='%Y-%j')
    output_df['month'].fillna(dates.dt.month, inplace=True)
    output_df['month'] = output_df['month'].astype(int)

    output_df.sort_values(['point_id', 'Year', 'month', 'doy'], inplace=True)
    output_df['scenario_id'] = scenario_id
    return output_df


def load_management_data(scenarios_file: str):
    """Load and pivot management schedule data."""
    scenarios_df = pd.read_csv(scenarios_file).rename({'simyear': 'Year'}, axis=1)
    
    # Extract scenario_id as string from 'scenario' column (e.g., 'scenario_8050' -> '8050')
    scenarios_df['scenario_id'] = scenarios_df['scenario'].str.replace('scenario_', '')
    
    scenarios_df = scenarios_df.pivot_table(
        index=['scenario_id', 'Year', 'doy'],
        columns='management',
        aggfunc='size',
        fill_value=0
    ).reset_index()
    
    return scenarios_df


def normalize_weather(weather_df: pd.DataFrame, train_pids: list):
    """Normalize weather data using StandardScaler fitted on training data."""
    train_weather = weather_df[weather_df['point_id'].isin(train_pids)].copy()
    scaler = StandardScaler()
    train_weather[['Tmax', 'Tmin', 'Precip']] = scaler.fit_transform(
        train_weather[['Tmax', 'Tmin', 'Precip']]
    )
    test_weather = weather_df[~weather_df['point_id'].isin(train_pids)].copy()
    test_weather[['Tmax', 'Tmin', 'Precip']] = scaler.transform(
        test_weather[['Tmax', 'Tmin', 'Precip']]
    )
    return pd.concat([train_weather, test_weather], ignore_index=True), scaler


def normalize_outputs(output_df: pd.DataFrame, train_pids: list, train_scenario_ids: list, train_years: list):
    """
    Normalize output variables (somsc, cgrain) using StandardScaler fitted on training data.
    
    Args:
        output_df: DataFrame with output data containing 'somsc' and 'cgrain' columns
        train_pids: List of training point IDs (strings)
        train_scenario_ids: List of training scenario IDs (strings like '8050')
        train_years: List of training years (integers)
    
    Returns:
        tuple: (normalized_output_df, scaler_Y)
    """
    # Separate train and test data
    train_mask = (output_df['point_id'].isin(train_pids)) & (output_df['scenario_id'].isin(train_scenario_ids)) & (output_df['Year'].isin(train_years))
    train_Y = output_df[train_mask].copy()
    test_Y = output_df[~train_mask].copy()

    # Fit scaler on training data
    scaler_Y = StandardScaler()
    train_Y[['somsc', 'cgrain']] = scaler_Y.fit_transform(train_Y[['somsc', 'cgrain']])
    
    # Transform test data
    test_Y[['somsc', 'cgrain']] = scaler_Y.transform(test_Y[['somsc', 'cgrain']])
    
    # Combine back together
    output_normalized = pd.concat([train_Y, test_Y], ignore_index=True)
    output_normalized.sort_values(['scenario_id', 'point_id', 'Year', 'doy'], inplace=True)
    
    print(f"Output normalization:")
    print(f"  Train samples: {len(train_Y)}")
    print(f"  Test samples:  {len(test_Y)}")
    print(f"  SOMSC - mean: {scaler_Y.mean_[0]:.4f}, std: {scaler_Y.scale_[0]:.4f}")
    print(f"  CGRAIN - mean: {scaler_Y.mean_[1]:.4f}, std: {scaler_Y.scale_[1]:.4f}")
    
    return output_normalized, scaler_Y


# ============================================================================
# LEGACY DATASET (Unchanged)
# ============================================================================

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
        
        # --- !! ADDED FOR VALIDATION !! ---
        # Store for easier comparison
        print("\n" + "="*80)
        print("LEGACY DATASET INFO (for validation)")
        # Make sure mapping is (sid, year, pid)
        # The user's __getitem__ implies this order:
        # sid, year, pid = self.mapping[idx]
        print(f"Mapping shape: {self.mapping.shape}")
        print(f"Sample mapping[0]: (sid={self.mapping[0,0]}, year={self.mapping[0,1]}, pid={self.mapping[0,2]})")
        print("="*80 + "\n")
        # --- !! END ADDED !! ---

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


# ============================================================================
# ***OPTIMIZED*** DYNAMIC DATASET (From Previous Turn)
# ============================================================================

class DayCentDatasetDynamic(Dataset):
    """
    Optimized flexible dataset with pre-computed indices for fast lookups
    and a virtual index mapping for massive scalability.
    
    This approach pre-computes all data-lookup indices during initialization
    for O(1) lookups in __getitem__ and uses math to resolve the 
    sample index (idx) to (scenario, year, point) without storing
    a massive list.
    
    IMPORTANT: All scenario identifiers must be strings (e.g., '8050', not 8050 or 'scenario_8050')
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
            "scenario_id": scenario_id
        }


# ============================================================================
# DATA PREPARATION HELPER (Unchanged)
# ============================================================================

def prepare_data_for_dynamic_dataset(base_dir, scenario_ids, train_config, val_config=None, test_config=None):
    """
    Load and normalize data for the dynamic dataset.
    
    Normalizes based on training set only (no data leakage).
    
    Args:
        base_dir: Base directory path
        scenario_ids: List of all scenario IDs to load
        train_config: Dict with 'scenarios', 'points', 'years' for training
        val_config: Optional dict with 'scenarios', 'points', 'years' for validation
        test_config: Optional dict with 'scenarios', 'points', 'years' for testing
    
    Returns:
        tuple: (weather_df, management_df, output_df, all_configs)
    """
    weather_dir = os.path.join(base_dir, "InputData/WeatherData")
    scenarios_file = os.path.join(base_dir, "InputData/schedule_scenarios_all_Synthetic_10000.csv")
    output_dir = os.path.join(base_dir, "OutputData_Synthetic_10000")
    
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # Load data
    print("\n1. Loading weather data...")
    weather_df = load_weather_data(weather_dir)
    print(f"   Weather data shape: {weather_df.shape}")
    
    print("\n2. Loading management data...")
    management_df = load_management_data(scenarios_file)
    print(f"   Management data shape: {management_df.shape}")
    
    print(f"\n3. Loading output data for {len(scenario_ids)} scenarios...")
    output_dfs = []
    max_workers = min(50, len(scenario_ids)) if len(scenario_ids) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_scenario_output, sid, output_dir): sid for sid in scenario_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading scenarios"):
            sid = futures[fut]
            try:
                df = fut.result()
                output_dfs.append(df)
            except Exception as e:
                print(f"Warning: failed to load scenario {sid}: {e}")

    if len(output_dfs) > 0:
        output_df = pd.concat(output_dfs, ignore_index=True)
    else:
        output_df = pd.DataFrame()
    print(f"   Output data shape: {output_df.shape}")
    
    # Get unique train points (for normalization - avoid duplicates)
    train_pids = list(set(train_config['points']))
    train_scenario_ids = [str(sid) for sid in train_config['scenarios']]
    train_years = train_config['years']
    
    print(f"\n4. Normalizing weather data...")
    print(f"   Using {len(train_pids)} unique training points for fitting")
    weather_df, weather_scaler = normalize_weather(weather_df, train_pids)
    
    print(f"\n5. Normalizing output data...")
    print(f"   Using {len(train_pids)} unique training points for fitting")
    output_df, output_scaler = normalize_outputs(output_df, train_pids, train_scenario_ids, train_years)
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    
    return weather_df, management_df, output_df, {
        'train': train_config,
        'val': val_config,
        'test': test_config,
        'weather_scaler': weather_scaler,
        'output_scaler': output_scaler
    }


# ============================================================================
# ***MODIFIED*** PROFILING & VALIDATION FUNCTION
# ============================================================================

def profile_datasets():
    """Profile and VALIDATE the performance of both dataset implementations."""
    import time
    import statistics
    
    print("\n" + "="*80)
    print("DATASET PERFORMANCE & VALIDATION")
    print("="*80)
    
    # Paths (Update these to your environment)
    base_dir = "/users/6/mehta423/daycent/data/SAS_KGML_090925"
    init_cond = os.path.join(base_dir, "InputData/initial_site_conditions.xlsx")
    
    # Legacy .npy paths (Update these)
    legacy_input_path = "/users/6/mehta423/daycent/data/experiment10/train_X.npy"
    legacy_output_path = "/users/6/mehta423/daycent/data/experiment10/train_Y.npy"
    
    
    # === !! NEW CONFIGURATION BASED ON YOUR DATA !! ===
    print("\nUsing specified train data for validation:")
    
    # 1. Scenarios:
    # scenario_1000, scenario_1099, ...
    scenarios_list = ['1000', '1099', '1390', '1616', '2034', '2600', '2729', '3503', '3932', '4880']
    
    # 2. Points:
    points_list = ['1773513', '1773749', '1773883', '1774528', '1774685', '1774853', '1775042',
 '1776558', '1776592', '1777257', '1777271', '1778109', '1778253', '1779437',
 '1779518', '1779588', '1782812', '1784153', '1784216', '1789230', '1790161',
 '1790233', '1790935', '1794265', '1794644', '1795246', '1795304', '1796415',
 '1798503', '657271', '657277', '657532', '657791', '658349', '659118', '659143',
 '659172', '661129', '661514', '663385', '664407', '664432', '664552', '664763',
 '665320', '666125', '667689', '668968', '669194', '669561', '669606', '669797',
 '670226', '670405', '670931', '671294', '672163', '674884', '675099', '675631',
 '676905', '677234', '678080', '678248', '678676', '679163', '679545', '680247',
 '681419', '681627', '681794', '682280', '683527', '684108', '684585', '684591',
 '685292', '685391', '685970', '686109', '686248', '687710', '688663', '689893',
 '690423', '691651', '693561', '693674', '694252', '696949', '697886', '697960',
 '698225', '699880', '700165', '700347', '701014', '701081', '701768', '703163',
 '708549', '710977']
    
    # 3. Years:
    years_list = list(range(2000, 2025)) # 2000 to 2024 inclusive
    
    train_config = {
        'scenarios': scenarios_list,
        'points': points_list,
        'years': years_list
    }
    
    print(f"\nConfiguration:")
    print(f"  Scenarios: {len(scenarios_list)}")
    print(f"  Points: {len(points_list)}")
    print(f"  Years: {len(years_list)}")
    print(f"  Expected samples: {len(scenarios_list) * len(points_list) * len(years_list)}")
    
    # ========================================================================
    # 1. LOAD LEGACY DATASET (DayCentDataset)
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING LEGACY DATASET (DayCentDataset)")
    print("="*80)
    
    legacy_dataset = None # Initialize
    if os.path.exists(legacy_input_path) and os.path.exists(legacy_output_path):
        print("\nCreating legacy dataset...")
        legacy_dataset = DayCentDataset(
            input_npy_path=legacy_input_path,
            output_npy_path=legacy_output_path,
            init_cond_path=init_cond,
            year_emb_dim=16
        )
        print(f"Legacy dataset size: {len(legacy_dataset)} samples")
    else:
        print(f"\nSkipping legacy dataset (files not found)")
        print(f"  Expected: {legacy_input_path}")
        
    
    # ========================================================================
    # 2. LOAD DYNAMIC DATASET (DayCentDatasetDynamic)
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING DYNAMIC DATASET (DayCentDatasetDynamic)")
    print("="*80)
    
    print("\nPreparing data (loading and normalizing)...")
    weather_df, management_df, output_df, configs = prepare_data_for_dynamic_dataset(
        base_dir=base_dir,
        scenario_ids=scenarios_list, # Load only the scenarios we need
        train_config=train_config,
        val_config=None,
        test_config=None
    )
    
    print("\nCreating dynamic dataset...")
    dynamic_dataset = DayCentDatasetDynamic(
        weather_df=weather_df,
        management_df=management_df,
        output_df=output_df,
        init_cond_path=init_cond,
        split_config=train_config,
        year_emb_dim=16
    )
    print(f"Dynamic dataset size: {len(dynamic_dataset)} samples")
    

    # ========================================================================
    # === !! 3. NEW VALIDATION STEP !! ===
    # ========================================================================
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    if legacy_dataset is not None and dynamic_dataset is not None:
        # Check 1: Length
        print(f"\nChecking length...")
        try:
            assert len(legacy_dataset) == len(dynamic_dataset)
            print(f"  ✓ SUCCESS: Both datasets have length {len(legacy_dataset)}")
        except AssertionError:
            print(f"  ✗ FAILURE: Length mismatch!")
            print(f"    Legacy:  {len(legacy_dataset)}")
            print(f"    Dynamic: {len(dynamic_dataset)}")
            return # Stop validation

        # Check 2: Random Item Comparison
        print(f"\nChecking content... (sampling 10 random indices)")
        indices_to_check = np.random.choice(len(legacy_dataset), 10, replace=False)
        all_match = True
        
        try:
            for idx in tqdm(indices_to_check, desc="Validating items"):
                legacy_item = legacy_dataset[idx]
                legacy_sid, legacy_year, legacy_pid = legacy_dataset.mapping[idx]
                
                dynamic_item = dynamic_dataset[idx]
                
                # A: Compare Keys (the most important check for index order)
                # Normalize legacy keys to match dynamic keys (all strings)
                
                # Handle scenario ID: "scenario_2729" -> "2729"
                sid_str = str(legacy_sid).replace('scenario_', '')
                
                # Handle year/pid: cast to int (to handle float), then back to str
                year_str = str(int(legacy_year))
                pid_str = str(int(legacy_pid))
                
                legacy_key = (sid_str, year_str, pid_str)
                dynamic_key = (dynamic_item['scenario_id'], dynamic_item['year'], dynamic_item['pid'])

                print(f"\nIndex {idx} Keys:")
                print(f"  Legacy:  {legacy_key}")
                print(f"  Dynamic: {dynamic_key}")

                assert legacy_key == dynamic_key, f"Key mismatch at index {idx}! Legacy: {legacy_key}, Dynamic: {dynamic_key}"
                
                # B: Compare non-sequence Tensors
                assert torch.allclose(legacy_item['init_cond'], dynamic_item['init_cond']), f"init_cond mismatch at index {idx}"
                assert torch.allclose(legacy_item['year_enc'], dynamic_item['year_enc']), f"year_enc mismatch at index {idx}"
                
                # C: Compare Output Tensors
                assert torch.allclose(legacy_item['somsc'], dynamic_item['somsc']), f"somsc mismatch at index {idx}"
                assert torch.allclose(legacy_item['somsc_mask'], dynamic_item['somsc_mask']), f"somsc_mask mismatch at index {idx}"
                assert torch.allclose(legacy_item['yield'], dynamic_item['yield']), f"yield mismatch at index {idx}"
                assert torch.allclose(legacy_item['yield_mask'], dynamic_item['yield_mask']), f"yield_mask mismatch at index {idx}"

            print("\n  ✓ SUCCESS: All 10 random samples match perfectly!")
            print("  (Keys, init_cond, year_enc, somsc, and yield all validated)")

        except AssertionError as e:
            print(f"\n  ✗ FAILURE: Data mismatch found!")
            print(f"    Error: {e}")
            all_match = False

    else:
        print("\nSkipping validation (Legacy dataset not loaded).")
        
        
    # ========================================================================
    # 4. PERFORMANCE PROFILING (Original)
    # ========================================================================
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING")
    print("="*80)
    
    legacy_mean = None
    if legacy_dataset:
        print("\nWarming up legacy dataset (10 samples)...")
        for i in range(min(10, len(legacy_dataset))):
            _ = legacy_dataset[i]
        
        num_samples = min(1000, len(legacy_dataset))
        print(f"\nProfiling legacy __getitem__ on {num_samples} samples...")
        
        times = []
        indices = np.random.choice(len(legacy_dataset), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="Legacy Dataset"):
            start = time.perf_counter()
            _ = legacy_dataset[idx]
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        print(f"\nLegacy Dataset Performance:")
        print(f"  Mean time:   {statistics.mean(times):.4f} ms")
        print(f"  Median time: {statistics.median(times):.4f} ms")
        legacy_mean = statistics.mean(times)
    
    # --- Profile Dynamic ---
    print("\nWarming up dynamic dataset (10 samples)...")
    for i in range(min(10, len(dynamic_dataset))):
        _ = dynamic_dataset[i]
    
    num_samples = min(1000, len(dynamic_dataset))
    print(f"\nProfiling dynamic __getitem__ on {num_samples} samples...")
    
    times = []
    indices = np.random.choice(len(dynamic_dataset), num_samples, replace=False)
    
    for idx in tqdm(indices, desc="Dynamic Dataset"):
        start = time.perf_counter()
        _ = dynamic_dataset[idx]
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    print(f"\nDynamic Dataset Performance:")
    print(f"  Mean time:   {statistics.mean(times):.4f} ms")
    print(f"  Median time: {statistics.median(times):.4f} ms")
    dynamic_mean = statistics.mean(times)
    
    # ========================================================================
    # 5. COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    if legacy_mean is not None:
        print(f"\nLegacy Dataset (pre-processed):  {legacy_mean:.4f} ms per sample")
        print(f"Dynamic Dataset (on-the-fly):    {dynamic_mean:.4f} ms per sample")
        
        if dynamic_mean < legacy_mean:
            speedup = legacy_mean / dynamic_mean
            print(f"  ✓ Dynamic dataset is {speedup:.2f}x FASTER")
        else:
            slowdown = dynamic_mean / legacy_mean
            print(f"  ✗ Dynamic dataset is {slowdown:.2f}x SLOWER")
    else:
        print(f"\nDynamic Dataset (on-the-fly):    {dynamic_mean:.4f} ms per sample")
        print("(Legacy dataset not available for performance comparison)")
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("DayCent Dataset Loader Performance Profiling & Validation")
    profile_datasets()