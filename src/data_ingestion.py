"""
Data Ingestion Module
=====================
Downloads NYC TLC Yellow Taxi trip data for 2019 and aggregates
ride counts by taxi zone and 30-minute time windows.
"""

import os
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import yaml
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    """Load project configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_taxi_data(config, force=False):
    """
    Download NYC TLC Yellow Taxi parquet files for all months in config.
    Files are saved to data/raw/.
    """
    raw_dir = config['data']['raw_dir']
    os.makedirs(raw_dir, exist_ok=True)
    
    year = config['data']['year']
    months = config['data']['months']
    base_url = config['data']['base_url']
    
    downloaded_files = []
    
    for month in months:
        url = base_url.format(month=month)
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join(raw_dir, filename)
        
        if os.path.exists(filepath) and not force:
            logger.info(f"Already exists: {filename}")
            downloaded_files.append(filepath)
            continue
        
        logger.info(f"Downloading: {filename}")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            downloaded_files.append(filepath)
            logger.info(f"Downloaded: {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
    
    return downloaded_files


def load_and_filter_month(filepath, config):
    """
    Load a single month's parquet file and extract relevant columns.
    Filters out invalid zone IDs and timestamps.
    """
    pickup_col = config['data']['pickup_col']
    zone_col = config['data']['zone_col']
    year = config['data']['year']
    
    # Read only needed columns
    df = pd.read_parquet(filepath, columns=[pickup_col, zone_col])
    
    # Rename for consistency
    df = df.rename(columns={pickup_col: 'pickup_datetime', zone_col: 'zone_id'})
    
    # Filter to valid dates within the target year
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])
    df = df[(df['pickup_datetime'].dt.year == year)]
    
    # Filter valid zone IDs (1-263 for NYC taxi zones)
    df = df.dropna(subset=['zone_id'])
    df = df[(df['zone_id'] >= 1) & (df['zone_id'] <= 263)]
    df['zone_id'] = df['zone_id'].astype(int)
    
    logger.info(f"  Loaded {len(df):,} valid trips from {os.path.basename(filepath)}")
    return df


def aggregate_demand(config, force=False):
    """
    Aggregate trip counts by zone and 30-minute time windows.
    
    Creates a complete grid of (zone x time_window) to ensure
    zero-demand periods are captured (important for modeling).
    
    Returns:
        DataFrame with columns: [pickup_datetime, zone_id, trip_count]
    """
    output_path = config['data']['aggregated_file']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path) and not force:
        logger.info(f"Loading existing aggregated data from {output_path}")
        return pd.read_parquet(output_path)
    
    raw_dir = config['data']['raw_dir']
    year = config['data']['year']
    window_min = config['data']['time_window_minutes']
    
    # Process each month
    all_trips = []
    for month in config['data']['months']:
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join(raw_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}, skipping.")
            continue
        
        df = load_and_filter_month(filepath, config)
        
        # Floor to 30-minute window
        df['time_window'] = df['pickup_datetime'].dt.floor(f'{window_min}min')
        
        # Count trips per zone per window
        counts = df.groupby(['time_window', 'zone_id']).size().reset_index(name='trip_count')
        all_trips.append(counts)
        
        del df
        gc.collect()
    
    if not all_trips:
        raise ValueError("No trip data loaded. Ensure raw data files exist.")
    
    demand = pd.concat(all_trips, ignore_index=True)
    
    # Aggregate again in case month boundaries overlap in windows
    demand = demand.groupby(['time_window', 'zone_id'])['trip_count'].sum().reset_index()
    
    # Create complete grid (all zones x all time windows) to fill zeros
    logger.info("Creating complete zone x time_window grid...")
    all_zones = np.arange(1, 264)  # Zones 1-263
    all_windows = pd.date_range(
        start=f'{year}-01-01',
        end=f'{year}-12-31 23:30:00',
        freq=f'{window_min}min'
    )
    
    full_index = pd.MultiIndex.from_product(
        [all_windows, all_zones],
        names=['time_window', 'zone_id']
    )
    full_grid = pd.DataFrame(index=full_index).reset_index()
    
    # Merge with actual counts
    demand = full_grid.merge(demand, on=['time_window', 'zone_id'], how='left')
    demand['trip_count'] = demand['trip_count'].fillna(0).astype(int)
    
    # Sort
    demand = demand.sort_values(['zone_id', 'time_window']).reset_index(drop=True)
    
    logger.info(f"Aggregated demand shape: {demand.shape}")
    logger.info(f"Time range: {demand['time_window'].min()} to {demand['time_window'].max()}")
    logger.info(f"Total zones: {demand['zone_id'].nunique()}")
    logger.info(f"Total trip count: {demand['trip_count'].sum():,}")
    
    # Save
    demand.to_parquet(output_path, index=False)
    logger.info(f"Saved aggregated demand to {output_path}")
    
    return demand


def download_zone_lookup(config):
    """Download taxi zone lookup table for zone names/boroughs."""
    url = config['data']['taxi_zone_lookup']
    output_path = os.path.join(config['data']['raw_dir'], 'taxi_zone_lookup.csv')
    
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    
    try:
        df = pd.read_csv(url)
        df.to_csv(output_path, index=False)
        logger.info(f"Downloaded taxi zone lookup ({len(df)} zones)")
        return df
    except Exception as e:
        logger.warning(f"Could not download zone lookup: {e}")
        return None


if __name__ == "__main__":
    config = load_config()
    
    # Step 1: Download raw data
    print("=" * 60)
    print("STEP 1: Downloading NYC TLC Yellow Taxi Data (2019)")
    print("=" * 60)
    download_taxi_data(config)
    download_zone_lookup(config)
    
    # Step 2: Aggregate demand
    print("\n" + "=" * 60)
    print("STEP 2: Aggregating demand by zone and 30-min windows")
    print("=" * 60)
    demand = aggregate_demand(config)
    
    print(f"\nDone! Aggregated {len(demand):,} records")
    print(f"Sample:\n{demand.head(10)}")
