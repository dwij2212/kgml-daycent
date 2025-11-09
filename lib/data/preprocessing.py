"""
Data preprocessing utilities for DayCent modeling.
"""
import os
import math
import numpy as np
import pandas as pd
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def load_weather_data(weather_dir: str) -> pd.DataFrame:
    """Load and concatenate all weather data files."""
    all_points = []
    
    for filename in os.listdir(weather_dir):
        if not filename.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(weather_dir, filename))
        df['point_id'] = filename.split(".csv")[0]
        all_points.append(df)
    
    weather_df = pd.concat(all_points, ignore_index=True)
    return weather_df


def split_by_quadrants(points_lookup_path: str, train_quadrants: list, test_quadrants: list):
    """Split points into train/test based on geographic quadrants."""
    df = pd.read_csv(points_lookup_path)
    
    # Calculate medians for splitting
    median_x = df['POINT_X'].median()
    median_y = df['POINT_Y'].median()
    
    # Create quadrants
    df['quadrant'] = 'Q1'
    df.loc[(df['POINT_X'] <= median_x) & (df['POINT_Y'] <= median_y), 'quadrant'] = 'Q1 (SW)'
    df.loc[(df['POINT_X'] > median_x) & (df['POINT_Y'] <= median_y), 'quadrant'] = 'Q2 (SE)'
    df.loc[(df['POINT_X'] <= median_x) & (df['POINT_Y'] > median_y), 'quadrant'] = 'Q3 (NW)'
    df.loc[(df['POINT_X'] > median_x) & (df['POINT_Y'] > median_y), 'quadrant'] = 'Q4 (NE)'
    
    train_checker = df[df['quadrant'].isin(train_quadrants)]
    test_checker = df[df['quadrant'].isin(test_quadrants)]
    
    print(f"Train ({', '.join(train_quadrants)}): {len(train_checker)} points")
    print(f"Test ({', '.join(test_quadrants)}): {len(test_checker)} points")
    
    train_pids = train_checker['id'].astype(str).unique()
    test_pids = test_checker['id'].astype(str).unique()
    
    return train_pids, test_pids


def normalize_weather_data(weather_df: pd.DataFrame, train_pids: list):
    """
    Normalize weather data using StandardScaler fitted on training data.
    
    Args:
        weather_df: DataFrame with weather data
        train_pids: List of training point IDs
    
    Returns:
        tuple: (normalized_weather_df, scaler)
    """
    train_weather = weather_df[weather_df['point_id'].isin(train_pids)].copy()
    test_weather = weather_df[~weather_df['point_id'].isin(train_pids)].copy()
    
    scaler = StandardScaler()
    train_weather[['Tmax', 'Tmin', 'Precip']] = scaler.fit_transform(
        train_weather[['Tmax', 'Tmin', 'Precip']]
    )
    test_weather[['Tmax', 'Tmin', 'Precip']] = scaler.transform(
        test_weather[['Tmax', 'Tmin', 'Precip']]
    )
    
    normalized_df = pd.concat([train_weather, test_weather], ignore_index=True)
    return normalized_df, scaler


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
    output_df.sort_index(inplace=True)

    output_df['scenario_id'] = scenario_id
    return output_df


def load_output_data(scenario_ids: list, output_dir: str, max_workers: int = None):
    """
    Load output data for multiple scenarios using multithreading.
    
    Args:
        scenario_ids: List of scenario IDs to load
        output_dir: Directory containing output CSV files
        max_workers: Number of threads for parallel loading
    
    Returns:
        pd.DataFrame: Concatenated output data with scenario_id column
    """
    all_outputs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_scenario = {
            executor.submit(load_single_scenario_output, scenario_id, output_dir): scenario_id 
            for scenario_id in scenario_ids
        }
        
        for future in tqdm(as_completed(future_to_scenario), 
                          desc="Loading scenarios", 
                          total=len(scenario_ids)):
            scenario_id = future_to_scenario[future]
            try:
                output_df = future.result()
                all_outputs.append(output_df)
            except Exception as exc:
                print(f'Scenario {scenario_id} error: {exc}')
    
    return pd.concat(all_outputs, ignore_index=True)


def load_management_data(scenario_ids: list, scenarios_file: str):
    """
    Load and pivot management schedule data.
    
    Args:
        scenarios_file: Path to scenarios CSV file
    
    Returns:
        pd.DataFrame: Pivoted management data with scenario_id column
    """
    scenarios_df = pd.read_csv(scenarios_file).rename({'simyear': 'Year'}, axis=1)
    
    scenarios_df['scenario_id'] = scenarios_df['scenario'].str.replace('scenario_', '')
    
    scenarios_df = scenarios_df.pivot_table(
        index=['scenario_id', 'Year', 'doy'],
        columns='management',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    scenarios_df = scenarios_df[scenarios_df['scenario_id'].isin(scenario_ids)]
    
    return scenarios_df


def load_data(scenario_ids: list, weather_df: pd.DataFrame, scenarios_file: str, 
              output_dir: str, max_workers: int = None):
    """Load and merge all data (management, weather, outputs)."""
    print("Loading management data...")
    management_df = load_management_data(scenario_ids, scenarios_file)

    # Create grid of all combinations
    scenarios = management_df["scenario"].unique()
    years = weather_df["Year"].unique()
    doys = weather_df["doy"].unique()

    grid = pd.DataFrame(itertools.product(scenarios, years, doys),
                        columns=["scenario", "Year", "doy"])

    grid_weather = pd.merge(grid, weather_df, on=["Year","doy"], how="left")

    X_daily = pd.merge(grid_weather, management_df, 
                    on=["scenario","Year","doy"], 
                    how="left")
    X_daily.fillna(0, inplace=True)
    X_daily = X_daily[X_daily['doy'] <= 365]
    X_daily.sort_values(['scenario', 'point_id', 'Year', 'doy'], inplace=True)
    X_daily.reset_index(drop=True, inplace=True)

    print("Loading output data...")
    Y = load_output_data(scenario_ids, output_dir, max_workers=max_workers)

    return X_daily, Y


def normalize_outputs(output_df: pd.DataFrame, train_pids: list, train_scenario_ids: list, 
                      train_years: list, scaler_path: str = None):
    """
    Normalize output variables (somsc, cgrain) using StandardScaler fitted on training data.
    
    Args:
        output_df: DataFrame with output data containing 'somsc' and 'cgrain' columns
        train_pids: List of training point IDs (strings)
        train_scenario_ids: List of training scenario IDs (strings like '8050')
        train_years: List of training years (integers)
        scaler_path: Optional path to save/load scaler
    
    Returns:
        tuple: (normalized_output_df, scaler_Y)
    """
    if scaler_path and os.path.exists(scaler_path):
        scaler_Y = joblib.load(scaler_path)
        print(f"Loaded existing scaler from {scaler_path}")
    else:
        train_mask = (
            (output_df['point_id'].isin(train_pids)) & 
            (output_df['scenario_id'].isin(train_scenario_ids)) & 
            (output_df['Year'].isin(train_years))
        )
        train_Y = output_df[train_mask].copy()
        
        scaler_Y = StandardScaler()
        scaler_Y.fit(train_Y[['somsc', 'cgrain']])
        
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler_Y, scaler_path)
            print(f"Saved new scaler to {scaler_path}")
        
        print(f"Output normalization fitted on {len(train_Y)} training samples")
        print(f"  SOMSC - mean: {scaler_Y.mean_[0]:.4f}, std: {scaler_Y.scale_[0]:.4f}")
        print(f"  CGRAIN - mean: {scaler_Y.mean_[1]:.4f}, std: {scaler_Y.scale_[1]:.4f}")
    
    train_mask = (
        (output_df['point_id'].isin(train_pids)) & 
        (output_df['scenario_id'].isin(train_scenario_ids)) & 
        (output_df['Year'].isin(train_years))
    )
    train_Y = output_df[train_mask].copy()
    test_Y = output_df[~train_mask].copy()
    
    train_Y[['somsc', 'cgrain']] = scaler_Y.transform(train_Y[['somsc', 'cgrain']])
    test_Y[['somsc', 'cgrain']] = scaler_Y.transform(test_Y[['somsc', 'cgrain']])
    
    output_normalized = pd.concat([train_Y, test_Y], ignore_index=True)
    output_normalized.sort_values(['scenario_id', 'point_id', 'Year', 'doy'], inplace=True)
    
    return output_normalized, scaler_Y


def create_and_save_sequences(X_daily: pd.DataFrame, Y: pd.DataFrame, pids: list, 
                               output_prefix: str, processed_dir: str):
    """Create sequences and save to .npy files."""
    print(f"\nCreating sequences for {output_prefix}...")
    
    df = X_daily[X_daily['point_id'].isin(pids)]

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in ["scenario", "Year", "point_id"]]

    # Group by scenario, year, and point_id
    groups = df.groupby(["scenario", "Year", "point_id"])

    # Create sequences and store mapping
    sequences = []
    mapping = []

    for (sid, year, pid), group in tqdm(groups, desc="Creating input sequences"):
        sequences.append(group[feature_cols].to_numpy())
        mapping.append((sid, year, pid))

    # Convert to arrays
    sequences = np.stack(sequences)
    mapping = np.array(mapping)

    # Save input sequences
    data_dict = {
        "data": sequences,
        "mapping": mapping,
        "columns": feature_cols
    }

    input_path = os.path.join(processed_dir, f"{output_prefix}_X.npy")
    np.save(input_path, data_dict, allow_pickle=True)
    print(f"Saved input sequences to {input_path}")

    # Process outputs
    Y_temp = Y[Y['point_id'].isin(pids)]
    Y_dict = {}
    for (sid, pid, year), group in tqdm(Y_temp.groupby(["scenario", "point_id", "Year"]), desc="Processing outputs"):
        Y_dict[(sid, pid, year)] = group[["month", "doy", "somsc", "cgrain"]].to_numpy()
    
    # Align Y to mapping
    somsc_list = []
    cgrain_list = []
    
    for sid, year, pid in tqdm(mapping, desc="Aligning outputs to mapping"):
        if (str(sid), str(pid), int(year)) in Y_dict:
            data = Y_dict[str(sid), str(pid), int(year)]

            somsc_array = np.full(12, np.nan, dtype=np.float64)
            
            # Filter valid somsc values
            valid_somsc_mask = ~np.isnan(data[:, 2])
            valid_data = data[valid_somsc_mask]
            
            if valid_data.size > 0:
                indices = valid_data[:, 0].astype(int) - 1
                values = valid_data[:, 2]
                somsc_array[indices] = values
            
            # Extract cgrain value
            cgrain_values = data[:, 3]
            valid_cgrain = cgrain_values[~np.isnan(cgrain_values)]
            
            if valid_cgrain.size > 0:
                cgrain_list.append(valid_cgrain[0])
                somsc_list.append(somsc_array)
            else:
                somsc_list.append(somsc_array)
                cgrain_list.append(np.nan)
        else:
            print(f"Warning: Missing data for point_id {pid}, Year {year}, scenario {sid}")
            somsc_list.append(np.full(12, np.nan, dtype=np.float64))
            cgrain_list.append(np.nan)

    # Convert to arrays
    final_somsc_array = np.array(somsc_list)
    final_cgrain_array = np.array(cgrain_list)

    print(f"\nOutput shapes: SOMSC {final_somsc_array.shape}, CGRAIN {final_cgrain_array.shape}")

    # Save outputs
    output_dict = {
        "somsc": final_somsc_array,
        "cgrain": final_cgrain_array,
    }

    output_path = os.path.join(processed_dir, f"{output_prefix}_Y.npy")
    np.save(output_path, output_dict, allow_pickle=True)
    print(f"Saved output sequences to {output_path}")


def prepare_experiment_data(config):
    """
    Complete data preparation pipeline for an experiment.
    
    Args:
        config: ExperimentConfig instance
    """
    from utils.config import ExperimentConfig
    
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig instance")
    
    print(f"\n{'='*60}")
    print(f"Preparing data for experiment: {config.experiment_id}")
    print(f"{'='*60}\n")
    
    # Load weather data
    print("Loading weather data...")
    weather_df = load_weather_data(config.data.weather_dir)
    
    # Split into train/test by quadrants
    print("\nSplitting data by quadrants...")
    train_pids, test_pids = split_by_quadrants(
        config.data.points_lookup,
        config.data.train_quadrants,
        config.data.test_quadrants
    )
    
    # Normalize weather
    print("\nNormalizing weather data...")
    weather_df, weather_scaler = normalize_weather_data(weather_df, train_pids, test_pids)
    
    # Load all data
    print(f"\nLoading data for {len(config.data.scenario_ids)} scenarios...")
    X_daily, Y = load_data(
        config.data.scenario_ids,
        weather_df,
        config.data.scenarios_file,
        config.data.output_dir,
        max_workers=config.data.max_workers
    )
    
    # Normalize outputs
    print("\nNormalizing output variables...")
    Y_normalized, scaler_Y = normalize_outputs(
        Y, train_pids, test_pids, 
        scaler_path=config.get_scaler_path()
    )
    
    # Create and save sequences
    print("\n" + "="*60)
    print("Creating and saving sequences...")
    print("="*60)
    
    create_and_save_sequences(X_daily, Y_normalized, train_pids, "train", config.processed_dir)
    create_and_save_sequences(X_daily, Y_normalized, test_pids, "test", config.processed_dir)
    
    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"Files saved to: {config.processed_dir}")
    print(f"{'='*60}\n")
    
    return {
        'train_pids': train_pids,
        'test_pids': test_pids,
        'weather_scaler': weather_scaler,
        'output_scaler': scaler_Y
    }
