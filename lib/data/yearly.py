"""
Data preparation and dataset utilities for yearly December SOMSC modeling.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_raw_data, normalize_outputs


MONTHLY_WEATHER_FEATURES = ["Tmax_mean", "Tmin_mean", "Precip_sum"]
MONTHLY_ID_COLUMNS = ["scenario_id", "point_id", "Year", "month"]


def _month_from_year_and_doy(years: pd.Series, doys: pd.Series) -> pd.Series:
    dates = pd.to_datetime(
        years.astype(str) + "-" + doys.astype(str),
        format="%Y-%j",
    )
    return dates.dt.month.astype(int)


def aggregate_weather_monthly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw daily weather to monthly features."""
    weather = weather_df.copy()
    weather["point_id"] = weather["point_id"].astype(str)
    weather["Year"] = weather["Year"].astype(int)
    weather["month"] = _month_from_year_and_doy(weather["Year"], weather["DOY"])

    monthly = (
        weather.groupby(["point_id", "Year", "month"], as_index=False)
        .agg(
            Tmax_mean=("Tmax", "mean"),
            Tmin_mean=("Tmin", "mean"),
            Precip_sum=("Precip", "sum"),
        )
        .sort_values(["point_id", "Year", "month"])
        .reset_index(drop=True)
    )
    return monthly


def normalize_monthly_weather_data(
    monthly_weather_df: pd.DataFrame,
    train_pids: list,
    train_years: list,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normalize monthly weather using train-point and train-year statistics."""
    weather = monthly_weather_df.copy()
    train_mask = weather["point_id"].isin([str(pid) for pid in train_pids])
    train_mask &= weather["Year"].isin(train_years)

    train_weather = weather.loc[train_mask, MONTHLY_WEATHER_FEATURES]
    if train_weather.empty:
        raise ValueError("No training rows available to fit the yearly weather scaler.")

    scaler = StandardScaler()
    scaler.fit(train_weather)
    weather[MONTHLY_WEATHER_FEATURES] = scaler.transform(
        weather[MONTHLY_WEATHER_FEATURES]
    )
    return weather, scaler


def aggregate_management_monthly(management_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate management events into monthly per-practice counts."""
    if "id" not in management_df.columns:
        raise ValueError(
            "Yearly SOMSC modeling requires point-specific management rows with an 'id' column."
        )

    management = management_df.copy()
    management["scenario_id"] = management["scenario_id"].astype(str)
    management["id"] = management["id"].astype(str)
    management["Year"] = management["Year"].astype(int)
    management["month"] = _month_from_year_and_doy(management["Year"], management["doy"])

    mgmt_cols = [
        col
        for col in management.columns
        if col not in ["scenario_id", "id", "Year", "doy", "month"]
    ]

    monthly = (
        management.groupby(["scenario_id", "id", "Year", "month"], as_index=False)[mgmt_cols]
        .sum()
        .rename(columns={"id": "point_id"})
        .sort_values(["scenario_id", "point_id", "Year", "month"])
        .reset_index(drop=True)
    )
    return monthly


def _load_initial_conditions(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).set_index("id")
    df.dropna(axis=1, inplace=True)
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


def prepare_yearly_data(config) -> Dict[str, Any]:
    """
    Load raw data and convert it into the monthly inputs needed by the yearly model.
    """
    print(f"\n{'=' * 80}")
    print(f"PREPARING YEARLY DATA: {config.experiment_id}")
    print(f"{'=' * 80}\n")

    train_config = config.data.get_train_config()
    if not train_config:
        raise ValueError("Training configuration must be specified for yearly preparation.")

    train_pids = train_config["points"]
    train_scenario_ids = train_config["scenarios"]
    train_years = train_config["years"]

    raw_data = load_raw_data(config)

    print("\n4. Aggregating monthly weather data...")
    weather_df = aggregate_weather_monthly(raw_data["weather_df"])
    print(f"   Monthly weather shape: {weather_df.shape}")

    print("\n5. Normalizing monthly weather data...")
    weather_df, weather_scaler = normalize_monthly_weather_data(
        weather_df, train_pids, train_years
    )

    print("\n6. Aggregating monthly management data...")
    management_df = aggregate_management_monthly(raw_data["management_df"])
    print(f"   Monthly management shape: {management_df.shape}")

    print("\n7. Normalizing outputs in the shared SOMSC space...")
    output_df, output_scaler = normalize_outputs(
        raw_data["output_df"].copy(),
        train_pids,
        train_scenario_ids,
        train_years,
        scaler_path=config.get_scaler_path(),
    )
    print(f"   Normalized output shape: {output_df.shape}")

    print("\n" + "=" * 80)
    print("YEARLY DATA PREPARATION COMPLETE")
    print("=" * 80 + "\n")

    return {
        "weather_df": weather_df,
        "management_df": management_df,
        "output_df": output_df,
        "weather_scaler": weather_scaler,
        "output_scaler": output_scaler,
    }


class YearlySOMSCDataset(Dataset):
    """
    Dataset for predicting December SOMSC from monthly weather and management aggregates.
    """

    def __init__(
        self,
        weather_df: pd.DataFrame,
        management_df: pd.DataFrame,
        output_df: pd.DataFrame,
        init_cond_path: str,
        split_config: Dict[str, Any],
        year_emb_dim: int = 16,
    ):
        if not split_config:
            raise ValueError("split_config must be provided")

        print("Initializing yearly SOMSC dataset...")

        self.year_emb_dim = year_emb_dim
        self.weather_cols = MONTHLY_WEATHER_FEATURES
        self.init_conditions = _load_initial_conditions(init_cond_path)
        self.mgmt_cols = [
            col for col in management_df.columns if col not in MONTHLY_ID_COLUMNS
        ]

        self.scenario_ids = [str(sid) for sid in split_config["scenarios"]]
        self.point_ids = [str(pid) for pid in split_config["points"]]
        self.years = [int(year) for year in split_config["years"]]

        self._build_lookup_indices(weather_df, management_df, output_df)
        self.samples = self._build_sample_index()
        self.total_len = len(self.samples)

        print(f"  Valid yearly samples: {self.total_len}")
        print(
            f"  Monthly features: {len(self.weather_cols)} weather + "
            f"{len(self.mgmt_cols)} management"
        )

    def _build_lookup_indices(
        self,
        weather_df: pd.DataFrame,
        management_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> None:
        weather = weather_df.copy()
        weather["point_id"] = weather["point_id"].astype(str)
        weather["Year"] = weather["Year"].astype(int)
        weather["month"] = weather["month"].astype(int)
        weather = weather.sort_values(["point_id", "Year", "month"])

        self.weather_index = {}
        for (pid, year), group in weather.groupby(["point_id", "Year"], sort=False):
            seq = np.zeros((12, len(self.weather_cols)), dtype=np.float32)
            months = group["month"].to_numpy(dtype=np.int32)
            vals = group[self.weather_cols].to_numpy(dtype=np.float32)
            valid = (months >= 1) & (months <= 12)
            seq[months[valid] - 1] = vals[valid]
            self.weather_index[(pid, int(year))] = seq

        management = management_df.copy()
        management["scenario_id"] = management["scenario_id"].astype(str)
        management["point_id"] = management["point_id"].astype(str)
        management["Year"] = management["Year"].astype(int)
        management["month"] = management["month"].astype(int)
        management = management.sort_values(["scenario_id", "point_id", "Year", "month"])

        self.mgmt_index = {}
        for (sid, pid, year), group in management.groupby(
            ["scenario_id", "point_id", "Year"], sort=False
        ):
            seq = np.zeros((12, len(self.mgmt_cols)), dtype=np.float32)
            months = group["month"].to_numpy(dtype=np.int32)
            vals = group[self.mgmt_cols].to_numpy(dtype=np.float32)
            valid = (months >= 1) & (months <= 12)
            seq[months[valid] - 1] = vals[valid]
            self.mgmt_index[(sid, pid, int(year))] = seq

        outputs = output_df.copy()
        outputs["scenario_id"] = outputs["scenario_id"].astype(str)
        outputs["point_id"] = outputs["point_id"].astype(str)
        outputs["Year"] = outputs["Year"].astype(int)
        outputs = outputs.sort_values(["scenario_id", "point_id", "Year", "month"])

        december = outputs[(outputs["month"] == 12) & outputs["somsc"].notna()].copy()
        self.december_somsc = {}
        for row in december.itertuples(index=False):
            key = (row.scenario_id, row.point_id, int(row.Year))
            self.december_somsc[key] = float(row.somsc)

    def _resolve_init_condition_key(self, pid: str) -> Optional[Any]:
        try:
            pid_int = int(pid)
            if pid_int in self.init_conditions.index:
                return pid_int
        except ValueError:
            pass

        if pid in self.init_conditions.index:
            return pid
        return None

    def _build_sample_index(self) -> list:
        samples = []
        missing_weather = 0
        missing_prev = 0
        missing_target = 0
        missing_init = 0

        for scenario_id in self.scenario_ids:
            for year in self.years:
                for pid in self.point_ids:
                    weather_key = (pid, year)
                    prev_key = (scenario_id, pid, year - 1)
                    target_key = (scenario_id, pid, year)

                    if weather_key not in self.weather_index:
                        missing_weather += 1
                        continue
                    if prev_key not in self.december_somsc:
                        missing_prev += 1
                        continue
                    if target_key not in self.december_somsc:
                        missing_target += 1
                        continue
                    if self._resolve_init_condition_key(pid) is None:
                        missing_init += 1
                        continue

                    samples.append((scenario_id, pid, year))

        print(
            "  Filtered invalid samples:"
            f" missing_weather={missing_weather},"
            f" missing_prev_dec={missing_prev},"
            f" missing_target_dec={missing_target},"
            f" missing_init={missing_init}"
        )
        return samples

    def _year_pos_enc(self, year: int) -> np.ndarray:
        year_rel = int(year) - 2000
        pe = np.zeros(self.year_emb_dim, dtype=np.float32)
        for idx in range(0, self.year_emb_dim, 2):
            div = np.power(10000, 2 * idx / self.year_emb_dim)
            pe[idx] = np.sin(year_rel / div)
            if idx + 1 < self.year_emb_dim:
                pe[idx + 1] = np.cos(year_rel / div)
        return pe

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.total_len}")

        scenario_id, pid, year = self.samples[idx]
        weather_seq = self.weather_index[(pid, year)]
        mgmt_seq = self.mgmt_index[(scenario_id, pid, year)]
        sequence = np.concatenate([weather_seq, mgmt_seq], axis=1).astype(np.float32)

        init_key = self._resolve_init_condition_key(pid)
        init_cond = self.init_conditions.loc[init_key].to_numpy(dtype=np.float32)
        prev_somsc = np.float32(self.december_somsc[(scenario_id, pid, year - 1)])
        target_somsc = np.float32(self.december_somsc[(scenario_id, pid, year)])
        year_pe = self._year_pos_enc(year)

        return {
            "sequence": torch.tensor(sequence, dtype=torch.float32),
            "init_cond": torch.tensor(init_cond, dtype=torch.float32),
            "year_enc": torch.tensor(year_pe, dtype=torch.float32),
            "somsc": torch.tensor(target_somsc, dtype=torch.float32),
            "somsc_mask": torch.tensor(1.0, dtype=torch.float32),
            "prev_somsc_state": torch.tensor(prev_somsc, dtype=torch.float32),
            "pid": pid,
            "year": str(year),
            "scenario_id": scenario_id,
        }


def create_yearly_dataset(
    prepared_data: Dict[str, Any],
    init_cond_path: str,
    split_config: Dict[str, Any],
    year_emb_dim: int = 16,
) -> YearlySOMSCDataset:
    return YearlySOMSCDataset(
        weather_df=prepared_data["weather_df"],
        management_df=prepared_data["management_df"],
        output_df=prepared_data["output_df"],
        init_cond_path=init_cond_path,
        split_config=split_config,
        year_emb_dim=year_emb_dim,
    )


def create_yearly_data_loaders(
    config,
    prepared_data: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], YearlySOMSCDataset]:
    train_config = config.data.get_train_config()
    val_config = config.data.get_val_config()
    test_config = config.data.get_test_config()

    if not train_config:
        raise ValueError("Training split configuration must be specified")

    if verbose:
        print("Creating yearly training dataset...")
    train_dataset = create_yearly_dataset(
        prepared_data=prepared_data,
        init_cond_path=config.data.init_cond_file,
        split_config=train_config,
        year_emb_dim=16,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )

    val_loader = None
    val_size = 0
    if val_config:
        if verbose:
            print("Creating yearly validation dataset...")
        val_dataset = create_yearly_dataset(
            prepared_data=prepared_data,
            init_cond_path=config.data.init_cond_file,
            split_config=val_config,
            year_emb_dim=16,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
        val_size = len(val_dataset)

    test_loader = None
    test_size = 0
    if test_config:
        if verbose:
            print("Creating yearly test dataset...")
        test_dataset = create_yearly_dataset(
            prepared_data=prepared_data,
            init_cond_path=config.data.init_cond_file,
            split_config=test_config,
            year_emb_dim=16,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
        test_size = len(test_dataset)

    if verbose:
        print("\nYearly dataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val:   {val_size}")
        print(f"  Test:  {test_size}")

    return train_loader, val_loader, test_loader, train_dataset
