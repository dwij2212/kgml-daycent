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
# DATA LOADING FUNCTIONS
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
# NEW DATASET (merging logic in __getitem__)
# ============================================================================

class DayCentDatasetDynamic(Dataset):
    """
    Flexible dataset that performs merging in __getitem__ with configurable train/val/test splits.
    
    This approach stores raw data and merges on-the-fly during iteration.
    Supports overlapping scenarios, points, and years across splits.
    
    IMPORTANT: All scenario identifiers must be strings (e.g., '8050', not 8050 or 'scenario_8050')
    """
    
    def __init__(self, weather_df, management_df, output_df, init_cond_path,
                 train_config=None, val_config=None, test_config=None,
                 dataset_type='train', year_emb_dim=16):
        """
        Args:
            weather_df: DataFrame with weather data (already normalized)
            management_df: DataFrame with management events (has 'scenario_id' column with strings)
            output_df: DataFrame with outputs (already normalized, has 'scenario_id' column with strings)
            init_cond_path: Path to initial conditions Excel file
            train_config: Dict with keys 'scenarios', 'points', 'years' (lists of strings/ints)
            val_config: Dict with keys 'scenarios', 'points', 'years' (lists of strings/ints)
            test_config: Dict with keys 'scenarios', 'points', 'years' (lists of strings/ints)
            dataset_type: One of 'train', 'val', 'test' - determines which mapping to use
            year_emb_dim: Dimension for year positional encoding
            
        Example config:
            train_config = {
                'scenarios': ['1', '2', '3'],  # or [1, 2, 3] - will be converted to strings
                'points': ['100', '101', '102'],
                'years': [2000, 2001, 2002]
            }
        """
        # Store all data (no filtering yet)
        self.weather_df = weather_df
        self.management_df = management_df
        self.output_df = output_df
        self.init_conditions = self._load_initial_conditions(init_cond_path)
        self.year_emb_dim = year_emb_dim
        self.dataset_type = dataset_type
        
        # Get management feature columns (exclude identifiers)
        self.mgmt_cols = [c for c in self.management_df.columns 
                         if c not in ['scenario_id', 'Year', 'doy']]
        
        # Create index mappings for each split
        self.train_mapping = []
        self.val_mapping = []
        self.test_mapping = []
        
        if train_config:
            self.train_mapping = self._create_index_mapping(
                train_config['scenarios'], 
                train_config['points'], 
                train_config['years']
            )
            print(f"Train mapping created: {len(self.train_mapping)} samples")
        
        if val_config:
            self.val_mapping = self._create_index_mapping(
                val_config['scenarios'], 
                val_config['points'], 
                val_config['years']
            )
            print(f"Val mapping created: {len(self.val_mapping)} samples")
        
        if test_config:
            self.test_mapping = self._create_index_mapping(
                test_config['scenarios'], 
                test_config['points'], 
                test_config['years']
            )
            print(f"Test mapping created: {len(self.test_mapping)} samples")
        
        # Set active mapping based on dataset_type
        self._set_dataset_type(dataset_type)
        
        print(f"\nDataset initialized in '{dataset_type}' mode with {len(self)} samples")
    
    def _create_index_mapping(self, scenario_ids, point_ids, years):
        """
        Create index mapping for given scenarios, points, and years.
        
        Args:
            scenario_ids: List of scenario IDs as strings (e.g., ['8050', '2765'])
            point_ids: List of point IDs as strings
            years: List of years as integers
            
        Returns:
            List of tuples (scenario_id, year, point_id) - all strings
        """
        mapping = []
        for scenario_id in scenario_ids:
            # Ensure scenario_id is string
            scenario_id = str(scenario_id)
            for year in years:
                for pid in point_ids:
                    mapping.append((scenario_id, year, str(pid)))
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
        else:  # test
            self.index_mapping = self.test_mapping
        
        if len(self.index_mapping) == 0:
            raise ValueError(f"No samples available for dataset_type '{dataset_type}'. "
                           f"Did you provide the corresponding config?")
    
    def set_mode(self, dataset_type):
        """
        Switch between train/val/test modes at runtime.
        
        Args:
            dataset_type: One of 'train', 'val', 'test'
        """
        self._set_dataset_type(dataset_type)
        print(f"Dataset mode changed to '{dataset_type}' with {len(self)} samples")

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
        return len(self.index_mapping)

    def __getitem__(self, idx):
        scenario_id, year, pid = self.index_mapping[idx]
        
        # All identifiers are now strings for consistency:
        # - scenario_id: e.g., '8050'
        # - year: e.g., 2000 (int from mapping)
        # - pid: e.g., '100'
        
        # 1. Get weather data for this point and year
        weather_data = self.weather_df[
            (self.weather_df['point_id'] == pid) & 
            (self.weather_df['Year'] == year)
        ].sort_values('doy')
        
        # 2. Get management data for this scenario and year
        mgmt_data = self.management_df[
            (self.management_df['scenario_id'] == scenario_id) & 
            (self.management_df['Year'] == year)
        ].sort_values('doy')
        
        # 3. Merge weather and management on doy
        # Create daily sequence (365 days)
        daily_df = pd.DataFrame({'doy': range(1, 366)})
        daily_df = daily_df.merge(weather_data[['doy', 'Tmax', 'Tmin', 'Precip']], 
                                  on='doy', how='left')
        daily_df = daily_df.merge(mgmt_data[['doy'] + self.mgmt_cols], 
                                  on='doy', how='left')
        daily_df.fillna(0, inplace=True)
        
        # 4. Create sequence array
        feature_cols = ['doy', 'Tmax', 'Tmin', 'Precip'] + self.mgmt_cols
        seq = daily_df[feature_cols].to_numpy().astype(np.float32)
        
        # 5. Process doy (sin/cos encoding)
        doy = seq[:, 0]
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        seq = np.concatenate([seq, doy_sin[:, None], doy_cos[:, None]], axis=1)
        
        # 6. Harvest mask
        harvest_col_idx = feature_cols.index('harvest_grain') if 'harvest_grain' in feature_cols else -1
        if harvest_col_idx >= 0:
            harvest_idx = np.where(seq[:, harvest_col_idx] == 1)[0]
            cutoff = harvest_idx[0] if len(harvest_idx) > 0 else 364
        else:
            cutoff = 364
        harvest_mask = np.zeros(365, dtype=np.float32)
        harvest_mask[:cutoff+1] = 1.0
        
        # 7. Initial site conditions
        init_cond = self.init_conditions.loc[int(pid)].to_numpy().astype(np.float32)
        
        # 8. Year encoding
        year_pe = self._year_pos_enc(year).astype(np.float32)
        
        # 9. Get outputs for this scenario, point, and year
        # scenario_id is already a string like '8050', use it directly
        output_data = self.output_df[
            (self.output_df['scenario_id'] == scenario_id) &
            (self.output_df['point_id'] == pid) &
            (self.output_df['Year'] == year)
        ]
        
        # 10. Process SOMSC (monthly values)
        somsc_array = np.full(12, np.nan, dtype=np.float32)
        if not output_data.empty:
            for _, row in output_data.iterrows():
                month = int(row['month'])
                if not pd.isna(row['somsc']):
                    somsc_array[month - 1] = row['somsc']
        
        somsc_mask = ~np.isnan(somsc_array)
        somsc_array = np.nan_to_num(somsc_array, nan=0.0)
        
        # 11. Process CGRAIN (annual value)
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
            "scenario_id": scenario_id  # Return as scenario_id (string) for consistency
        }


# ============================================================================
# DATA PREPARATION HELPER
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


def test_flexible_dataset():
    """Test the flexible dynamic dataset with train/val/test splits."""
    print("\n" + "="*80)
    print("TESTING FLEXIBLE DYNAMIC DATASET")
    print("="*80)
    
    # Paths
    base_dir = "/users/6/mehta423/daycent/data/SAS_KGML_090925"
    init_cond = os.path.join(base_dir, "InputData/initial_site_conditions.xlsx")
    points_lookup = os.path.join(base_dir, "SAS_points_lookup.csv")
    
    # Get point IDs
    points_df = pd.read_csv(points_lookup)
    all_pids = points_df['id'].astype(str).tolist()
    
    # Example: Create flexible train/val/test configurations
    # These can overlap! Same scenario/point/year can be in multiple splits
    
    # For demo, let's use a subset
    test_scenario_ids = [str(i) for i in range(1, 1000)]
    
    # Split points into groups (can overlap if desired)
    train_points = all_pids[:100]  # First 100 points
    val_points = all_pids[100:150]  # 100-150 (overlaps with train!)
    test_points = all_pids[150:]  # 150-200 (overlaps with val!)
    
    # Years can also overlap
    train_years = list(range(2000, 2020))  # 2000-2019
    val_years = list(range(2008, 2020))    # 2008-2019 (overlaps!)
    test_years = list(range(2020, 2024))   # 2020-2023 (overlaps!)

    # Scenarios can overlap too
    train_scenarios = ['99', '2765', '6111', '2425', '7695', '2795', '1773', '1803', '9017', '1639', '9109', '1087', '368', '7221', '6305', '8457', '3396', '7727', '5211', '7759', '6563', '9387', '9223', '8180', '5142', '5003', '5533', '9169', '4947', '9319', '2670', '8670', '1273', '5624', '4361', '7737', '4770', '5967', '534', '6909', '6232', '5848', '143', '4283', '7900', '4618', '5089', '9568', '9277', '4006']
    val_scenarios = ['1', '10', '100', '1000', '10000', '1001', '2002', '1003', '1004', '1005']
    test_scenarios = ['2425', '7695']
    
    train_config = {
        'scenarios': train_scenarios,
        'points': train_points,
        'years': train_years
    }
    
    val_config = {
        'scenarios': val_scenarios,
        'points': val_points,
        'years': val_years
    }
    
    test_config = {
        'scenarios': test_scenarios,
        'points': test_points,
        'years': test_years
    }
    
    print(f"\nConfiguration:")
    print(f"  Train: {len(train_scenarios)} scenarios, {len(train_points)} points, {len(train_years)} years")
    print(f"  Val:   {len(val_scenarios)} scenarios, {len(val_points)} points, {len(val_years)} years")
    print(f"  Test:  {len(test_scenarios)} scenarios, {len(test_points)} points, {len(test_years)} years")
    
    # Prepare data (loads and normalizes based on training set)
    weather_df, management_df, output_df, configs = prepare_data_for_dynamic_dataset(
        base_dir=base_dir,
        scenario_ids=test_scenario_ids,
        train_config=train_config,
        val_config=val_config,
        test_config=test_config
    )
    
    # Create dataset (starts in 'train' mode)
    print("\n" + "="*80)
    print("CREATING FLEXIBLE DATASET")
    print("="*80)
    
    dataset = DayCentDatasetDynamic(
        weather_df=weather_df,
        management_df=management_df,
        output_df=output_df,
        init_cond_path=init_cond,
        train_config=train_config,
        val_config=val_config,
        test_config=test_config,
        dataset_type='train',  # Start in train mode
        year_emb_dim=16
    )
    
    # Test switching modes
    print("\n" + "="*80)
    print("TESTING MODE SWITCHING")
    print("="*80)
    
    print(f"\nInitial mode: train - {len(dataset)} samples")
    train_sample = dataset[0]
    print(f"Train sample 0: scenario_id={train_sample['scenario_id']}, pid={train_sample['pid']}, year={train_sample['year']}")
    
    # Switch to validation
    dataset.set_mode('val')
    val_sample = dataset[0]
    print(f"Val sample 0: scenario_id={val_sample['scenario_id']}, pid={val_sample['pid']}, year={val_sample['year']}")
    
    # Switch to test
    dataset.set_mode('test')
    test_sample = dataset[0]
    print(f"Test sample 0: scenario_id={test_sample['scenario_id']}, pid={test_sample['pid']}, year={test_sample['year']}")
    
    # Switch back to train
    dataset.set_mode('train')
    print(f"\nSwitched back to train mode - {len(dataset)} samples")
    
    print("\n" + "="*80)
    print("SAMPLE STRUCTURE")
    print("="*80)
    for key, val in train_sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:15s}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key:15s}: {val}")
    
    print("\n" + "="*80)
    print("SUCCESS! The flexible dataset works correctly.")
    print("You can now:")
    print("  1. Define overlapping train/val/test splits")
    print("  2. Switch between modes at runtime with dataset.set_mode()")
    print("  3. Normalization is done only on training data (no leakage)")
    print("="*80)
    
    return dataset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("DayCent Dataset Loader Testing")

    test_flexible_dataset()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
