"""
Data loading utilities for DayCent experiments.

This module provides utilities for creating data loaders and datasets
from experiment configuration.
"""
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader

from .dataset import DayCentDatasetV2


def create_dataset(prepared_data: Dict[str, Any], 
                   init_cond_path: str,
                   split_config: Dict[str, Any],
                   year_emb_dim: int = 16) -> DayCentDatasetV2:
    """
    Create a DayCentDatasetV2 instance.
    
    Args:
        prepared_data: Dict with 'weather_df', 'management_df', 'output_df'
        init_cond_path: Path to initial conditions file
        split_config: Split configuration dict with 'scenarios', 'points', 'years'
        year_emb_dim: Dimension for year positional encoding
        
    Returns:
        DayCentDatasetV2 instance
    """
    return DayCentDatasetV2(
        weather_df=prepared_data['weather_df'],
        management_df=prepared_data['management_df'],
        output_df=prepared_data['output_df'],
        init_cond_path=init_cond_path,
        split_config=split_config,
        year_emb_dim=year_emb_dim
    )


def create_data_loaders(config, prepared_data: Dict[str, Any], 
                        verbose: bool = True) -> Tuple[DataLoader, Optional[DataLoader], 
                                                        Optional[DataLoader], DayCentDatasetV2]:
    """
    Create train/val/test data loaders using DayCentDatasetV2.
    
    Creates separate dataset instances for each split since DataLoader holds 
    references (not copies) to dataset objects.
    
    Args:
        config: ExperimentConfig instance
        prepared_data: Dict with 'weather_df', 'management_df', 'output_df' 
                       from prepare_data_for_datasetv2
        verbose: Whether to print dataset sizes
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset)
    """
    # Get split configurations
    train_config = config.data.get_train_config()
    val_config = config.data.get_val_config()
    test_config = config.data.get_test_config()
    
    if not train_config:
        raise ValueError("Training split configuration must be specified")
    
    # Create training dataset
    if verbose:
        print("Creating training dataset...")
    train_dataset = create_dataset(
        prepared_data=prepared_data,
        init_cond_path=config.data.init_cond_file,
        split_config=train_config,
        year_emb_dim=16
    )
    
    train_size = len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    # Create validation dataset
    val_loader = None
    val_size = 0
    if val_config:
        if verbose:
            print("Creating validation dataset...")
        val_dataset = create_dataset(
            prepared_data=prepared_data,
            init_cond_path=config.data.init_cond_file,
            split_config=val_config,
            year_emb_dim=16
        )
        val_size = len(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
    
    # Create test dataset
    test_loader = None
    test_size = 0
    if test_config:
        if verbose:
            print("Creating test dataset...")
        test_dataset = create_dataset(
            prepared_data=prepared_data,
            init_cond_path=config.data.init_cond_file,
            split_config=test_config,
            year_emb_dim=16
        )
        test_size = len(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
    
    if verbose:
        print(f"\nDataset sizes:")
        print(f"  Train: {train_size}")
        print(f"  Val:   {val_size}")
        print(f"  Test:  {test_size}")
    
    return train_loader, val_loader, test_loader, train_dataset
