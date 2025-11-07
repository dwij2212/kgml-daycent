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
    
    scenarios_df = scenarios_df.pivot_table(
        index=['scenario', 'Year', 'doy'],
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


def normalize_outputs(output_df: pd.DataFrame, train_pids: list):
    """
    Normalize output variables (somsc, cgrain) using StandardScaler fitted on training data.
    
    Args:
        output_df: DataFrame with output data containing 'somsc' and 'cgrain' columns
        train_pids: List of training point IDs
    
    Returns:
        tuple: (normalized_output_df, scaler_Y)
    """
    # Separate train and test data
    train_Y = output_df[output_df['point_id'].isin(train_pids)].copy()
    test_Y = output_df[~output_df['point_id'].isin(train_pids)].copy()
    
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
# ORIGINAL DATASET (uses preprocessed .npy files)
# ============================================================================

class DayCentDatasetOriginal(Dataset):
    """Original dataset that loads from preprocessed .npy files."""
    
    def __init__(self, input_npy_path, output_npy_path, init_cond_path, year_emb_dim=16):
        # Load preprocessed input sequences
        data_dict = np.load(input_npy_path, allow_pickle=True).item()
        self.data = data_dict["data"]        # (N, 365, #features)
        self.mapping = data_dict["mapping"]  # (N, 3) => (scenario, year, point_id)
        self.columns = list(data_dict["columns"])

        # Load initial conditions
        self.init_conditions = self._load_initial_conditions(init_cond_path)
        self.year_emb_dim = year_emb_dim

        # Load preprocessed outputs
        data_dict = np.load(output_npy_path, allow_pickle=True).item()
        self.somsc = data_dict["somsc"]
        self.cgrain = data_dict["cgrain"]

    def _load_initial_conditions(self, path):
        df = pd.read_excel(path).set_index("id")
        df.dropna(axis=1, inplace=True)
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

        # Harvest mask
        harvest_idx = np.where(seq[:, self.columns.index("harvest_grain")] == 1)[0]
        cutoff = harvest_idx[0] if len(harvest_idx) > 0 else 364
        harvest_mask = np.zeros(365, dtype=np.float32)
        harvest_mask[:cutoff+1] = 1.0

        # Process doy (sin/cos encoding)
        doy_idx = self.columns.index("doy")
        doy = seq[:, doy_idx]
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        seq = np.concatenate([seq, doy_sin[:, None], doy_cos[:, None]], axis=1)

        # Initial site conditions
        init_cond = self.init_conditions.loc[int(pid)].to_numpy().astype(np.float32)

        # Year encoding
        year_pe = self._year_pos_enc(year).astype(np.float32)

        # Labels
        somsc = self.somsc[idx]
        somsc_mask = ~np.isnan(somsc)
        somsc = np.nan_to_num(somsc, nan=0.0)

        yield_val = self.cgrain[idx]
        yield_mask = ~np.isnan(yield_val)
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
            "year": year,
            "scenario": sid
        }


# ============================================================================
# NEW DATASET (merging logic in __getitem__)
# ============================================================================

class DayCentDatasetDynamic(Dataset):
    """
    Experimental dataset that performs merging in __getitem__ instead of preprocessing.
    
    This approach stores raw data and merges on-the-fly during iteration.
    """
    
    def __init__(self, weather_df, management_df, output_df, init_cond_path, 
                 point_ids, scenario_ids, year_emb_dim=16):
        """
        Args:
            weather_df: DataFrame with weather data (point_id, Year, doy, Tmax, Tmin, Precip)
            management_df: DataFrame with management events (scenario, Year, doy, event_columns)
            output_df: DataFrame with outputs (scenario_id, point_id, Year, doy/month, somsc, cgrain)
            init_cond_path: Path to initial conditions Excel file
            point_ids: List of point IDs to include in this dataset
            scenario_ids: List of scenario IDs to include
            year_emb_dim: Dimension for year positional encoding
        """
        self.weather_df = weather_df[weather_df['point_id'].isin(point_ids)].copy()
        self.management_df = management_df[
            management_df['scenario'].isin([f'scenario_{sid}' for sid in scenario_ids])
        ].copy()
        self.output_df = output_df[
            (output_df['point_id'].isin(point_ids)) & 
            (output_df['scenario_id'].isin([str(s) for s in scenario_ids]))
        ].copy()
        
        self.init_conditions = self._load_initial_conditions(init_cond_path)
        self.year_emb_dim = year_emb_dim
        
        # Get unique years and scenarios
        self.years = sorted(self.weather_df['Year'].unique())
        self.scenarios = sorted(self.management_df['scenario'].unique())
        self.point_ids = sorted(point_ids)
        
        # Create index mapping (scenario, year, point_id) -> idx
        self.index_mapping = []
        for scenario in self.scenarios:
            for year in self.years:
                for pid in self.point_ids:
                    self.index_mapping.append((scenario, year, pid))
        
        # Get management feature columns (exclude identifiers)
        self.mgmt_cols = [c for c in self.management_df.columns 
                         if c not in ['scenario', 'Year', 'doy']]
        
        print(f"Dataset initialized:")
        print(f"  Scenarios: {len(self.scenarios)}")
        print(f"  Years: {len(self.years)}")
        print(f"  Points: {len(self.point_ids)}")
        print(f"  Total samples: {len(self.index_mapping)}")

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
        scenario, year, pid = self.index_mapping[idx]
        
        # 1. Get weather data for this point and year
        weather_data = self.weather_df[
            (self.weather_df['point_id'] == pid) & 
            (self.weather_df['Year'] == year)
        ].sort_values('doy')
        
        # 2. Get management data for this scenario and year
        mgmt_data = self.management_df[
            (self.management_df['scenario'] == scenario) & 
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
        scenario_id = scenario.split('_')[1]  # Extract ID from 'scenario_X'
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
            "scenario": scenario
        }


# ============================================================================
# TESTING AND COMPARISON
# ============================================================================

def test_original_dataset():
    """Test the original dataset loader."""
    print("\n" + "="*80)
    print("TESTING ORIGINAL DATASET (preprocessed .npy files)")
    print("="*80)
    
    # Paths
    input_npy = "/users/6/mehta423/daycent/data/experiment10/train_50_X.npy"
    output_npy = "/users/6/mehta423/daycent/data/experiment10/train_50_Y.npy"
    init_cond = "/users/6/mehta423/daycent/data/SAS_KGML_090925/InputData/initial_site_conditions.xlsx"
    
    # Create dataset
    dataset = DayCentDatasetOriginal(input_npy, output_npy, init_cond)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample 0 structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:15s}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key:15s}: {val}")
    
    return dataset, sample


def test_dynamic_dataset():
    """Test the new dynamic dataset loader."""
    print("\n" + "="*80)
    print("TESTING DYNAMIC DATASET (merging in __getitem__)")
    print("="*80)
    
    # Paths
    base_dir = "/users/6/mehta423/daycent/data/SAS_KGML_090925"
    weather_dir = os.path.join(base_dir, "InputData/WeatherData")
    scenarios_file = os.path.join(base_dir, "InputData/schedule_scenarios_all_Synthetic_10000.csv")
    output_dir = os.path.join(base_dir, "OutputData_Synthetic_10000")
    init_cond = os.path.join(base_dir, "InputData/initial_site_conditions.xlsx")
    points_lookup = os.path.join(base_dir, "SAS_points_lookup.csv")
    
    # Load data
    print("\n1. Loading weather data...")
    weather_df = load_weather_data(weather_dir)
    print(f"   Weather data shape: {weather_df.shape}")
    
    print("\n2. Loading management data...")
    management_df = load_management_data(scenarios_file)
    print(f"   Management data shape: {management_df.shape}")
    
    # For testing, use just a few scenarios
    test_scenario_ids = [8050, 2765, 6111, 2425, 7695, 2795, 1773, 1803, 9017, 1639, 9109, 1087, 368, 7221, 6305, 8457, 3396, 7727, 5211, 7759, 6563, 9387, 9223, 8180, 5142, 5003, 5533, 9169, 4947, 9319, 2670, 8670, 1273, 5624, 4361, 7737, 4770, 5967, 534, 6909, 6232, 5848, 143, 4283, 7900, 4618, 5089, 9568, 9277, 4006]
    test_scenario_ids = [str(sid) for sid in test_scenario_ids]
    print(f"\n3. Loading output data for {len(test_scenario_ids)} scenarios...")
    output_dfs = []
    for sid in test_scenario_ids:
        output_df = load_single_scenario_output(sid, output_dir)
        output_dfs.append(output_df)
    output_df = pd.concat(output_dfs, ignore_index=True)
    print(f"   Output data shape: {output_df.shape}")
    
    print("\n4. Getting train point IDs...")
    points_df = pd.read_csv(points_lookup)
    median_x = points_df['POINT_X'].median()
    median_y = points_df['POINT_Y'].median()
    train_points = points_df[
        (points_df['POINT_X'] <= median_x) & 
        (points_df['POINT_Y'] <= median_y)
    ]
    train_pids = train_points['id'].astype(str).tolist()
    print(f"   Train points: {len(train_pids)}")
    
    print("\n5. Normalizing weather data...")
    weather_df, _ = normalize_weather(weather_df, train_pids)
    
    print("\n6. Normalizing output data...")
    output_df, scaler_Y = normalize_outputs(output_df, train_pids)
    
    print("\n7. Creating dynamic dataset...")
    dataset = DayCentDatasetDynamic(
        weather_df=weather_df,
        management_df=management_df,
        output_df=output_df,
        init_cond_path=init_cond,
        point_ids=train_pids,
        scenario_ids=test_scenario_ids,
        year_emb_dim=16
    )
    
    print(f"\n8. Getting sample...")
    sample = dataset[0]
    
    print(f"\nSample 0 structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:15s}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key:15s}: {val}")
    
    return dataset, sample


def compare_datasets():
    """Compare outputs from both dataset implementations."""
    print("\n" + "="*80)
    print("COMPARING DATASET IMPLEMENTATIONS")
    print("="*80)
    
    # Test original
    orig_dataset, orig_sample = test_original_dataset()
    
    # Test dynamic
    dyn_dataset, dyn_sample = test_dynamic_dataset()
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nOriginal dataset size: {len(orig_dataset)}")
    print(f"Dynamic dataset size:  {len(dyn_dataset)}")
    
    print(f"\nSequence shapes:")
    print(f"  Original: {orig_sample['sequence'].shape}")
    print(f"  Dynamic:  {dyn_sample['sequence'].shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Original SOMSC: {orig_sample['somsc'].shape}")
    print(f"  Dynamic SOMSC:  {dyn_sample['somsc'].shape}")
    print(f"  Original Yield: {orig_sample['yield'].shape}")
    print(f"  Dynamic Yield:  {dyn_sample['yield'].shape}")
    
    # ========================================================================
    # VALUE COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("VALUE EQUALITY TESTING")
    print("="*80)
    
    def compare_tensors(name, tensor1, tensor2, rtol=1e-3, atol=1e-4):
        """Compare two tensors and report differences."""
        if tensor1.shape != tensor2.shape:
            print(f"\n❌ {name}: SHAPES DIFFER")
            print(f"   Original: {tensor1.shape}, Dynamic: {tensor2.shape}")
            return False
        
        # Check for exact equality
        exact_match = torch.equal(tensor1, tensor2)
        if exact_match:
            print(f"\n✅ {name}: EXACT MATCH")
            return True
        
        # Check for close equality (accounting for floating point errors)
        close_match = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        if close_match:
            print(f"\n✅ {name}: CLOSE MATCH (within tolerance)")
            max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            print(f"   Max difference: {max_diff:.2e}")
            return True
        
        # Report differences
        print(f"\n❌ {name}: VALUES DIFFER")
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        num_diff = torch.sum(~torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol)).item()
        total = tensor1.numel()
        
        print(f"   Max difference:  {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Differing values: {num_diff}/{total} ({100*num_diff/total:.2f}%)")
        
        # Show some example differences
        if num_diff > 0:
            flat_t1 = tensor1.flatten()
            flat_t2 = tensor2.flatten()
            flat_diff = diff.flatten()
            
            # Get indices of largest differences
            top_diff_indices = torch.topk(flat_diff, min(5, num_diff)).indices
            print(f"   Top 5 differences:")
            for i, idx in enumerate(top_diff_indices):
                print(f"     {i+1}. Original: {flat_t1[idx]:.6f}, Dynamic: {flat_t2[idx]:.6f}, Diff: {flat_diff[idx]:.6f}")
        
        return False
    
    # Compare each field
    all_match = True
    
    # Sequence comparison
    all_match &= compare_tensors("Sequence", orig_sample['sequence'], dyn_sample['sequence'])
    
    # Initial conditions comparison
    all_match &= compare_tensors("Initial Conditions", orig_sample['init_cond'], dyn_sample['init_cond'])
    
    # Year encoding comparison
    all_match &= compare_tensors("Year Encoding", orig_sample['year_enc'], dyn_sample['year_enc'])
    
    # SOMSC comparison
    all_match &= compare_tensors("SOMSC", orig_sample['somsc'], dyn_sample['somsc'])
    
    # SOMSC mask comparison
    all_match &= compare_tensors("SOMSC Mask", orig_sample['somsc_mask'], dyn_sample['somsc_mask'])
    
    # Yield comparison
    all_match &= compare_tensors("Yield", orig_sample['yield'].unsqueeze(0), dyn_sample['yield'].unsqueeze(0))
    
    # Yield mask comparison
    all_match &= compare_tensors("Yield Mask", orig_sample['yield_mask'].unsqueeze(0), dyn_sample['yield_mask'].unsqueeze(0))
    
    # Harvest mask comparison
    all_match &= compare_tensors("Harvest Mask", orig_sample['harvest_mask'], dyn_sample['harvest_mask'])
    
    # Metadata comparison
    print(f"\n{'='*80}")
    print("METADATA COMPARISON")
    print(f"{'='*80}")
    print(f"Point ID - Original: {orig_sample['pid']}, Dynamic: {dyn_sample['pid']}")
    print(f"Year     - Original: {orig_sample['year']}, Dynamic: {dyn_sample['year']}")
    
    pid_match = str(orig_sample['pid']) == str(dyn_sample['pid'])
    year_match = str(orig_sample['year']) == str(dyn_sample['year'])
    
    if pid_match and year_match:
        print("✅ Metadata matches")
    else:
        print("❌ Metadata differs")
        all_match = False
    
    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    if all_match:
        print("✅ ALL VALUES MATCH - Implementations are equivalent!")
    else:
        print("❌ SOME VALUES DIFFER - Check the differences above")
    
    return all_match


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("DayCent Dataset Loader Testing")
    print("="*80)
    print("\nThis script allows you to test different dataset loading approaches:")
    print("1. Original: Uses preprocessed .npy files")
    print("2. Dynamic: Merges data on-the-fly in __getitem__")
    print("\nYou can modify and experiment with the DayCentDatasetDynamic class")
    print("to test your own implementation ideas.")
    
    # Run comparison
    # test_original_dataset()
    # test_dynamic_dataset()
    compare_datasets()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
