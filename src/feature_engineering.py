"""
Feature Engineering Module
==========================
Creates temporal, lag, and rolling window features for ride demand prediction.
All features are zone-specific to capture local patterns.
"""

import pandas as pd
import numpy as np
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_temporal_features(df):
    """
    Add time-based features derived from the time window.
    
    Features:
        - hour: Hour of day (0-23)
        - minute_30: Whether it's the :30 half-hour (0 or 1)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Binary weekend indicator
        - month: Month of year (1-12)
        - day_of_month: Day of month (1-31)
        - week_of_year: Week number (1-52)
        - is_rush_hour: Binary flag for typical rush hours (7-9 AM, 5-7 PM weekdays)
        - is_night: Binary flag for late night hours (11 PM - 5 AM)
        - time_sin, time_cos: Cyclical encoding of time of day
        - dow_sin, dow_cos: Cyclical encoding of day of week
    """
    dt = df['time_window']
    
    df['hour'] = dt.dt.hour
    df['minute_30'] = (dt.dt.minute >= 30).astype(int)
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = dt.dt.month
    df['day_of_month'] = dt.dt.day
    df['week_of_year'] = dt.dt.isocalendar().week.values.astype(int)
    
    # Rush hour: 7-9 AM and 5-7 PM on weekdays
    df['is_rush_hour'] = (
        (df['is_weekend'] == 0) & 
        (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
         ((df['hour'] >= 17) & (df['hour'] <= 19)))
    ).astype(int)
    
    # Night hours: 11 PM - 5 AM
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    
    # Cyclical encoding of time (period = 48 half-hours = 24 hours)
    half_hour_of_day = df['hour'] * 2 + df['minute_30']
    df['time_sin'] = np.sin(2 * np.pi * half_hour_of_day / 48)
    df['time_cos'] = np.cos(2 * np.pi * half_hour_of_day / 48)
    
    # Cyclical encoding of day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # US Federal Holidays for 2019 (significantly affect NYC taxi demand)
    us_holidays_2019 = pd.to_datetime([
        '2019-01-01',  # New Year's Day
        '2019-01-21',  # MLK Day
        '2019-02-18',  # Presidents' Day
        '2019-05-27',  # Memorial Day
        '2019-07-04',  # Independence Day
        '2019-09-02',  # Labor Day
        '2019-10-14',  # Columbus Day
        '2019-11-11',  # Veterans Day
        '2019-11-28',  # Thanksgiving
        '2019-12-25',  # Christmas
    ])
    df['is_holiday'] = df['time_window'].dt.date.isin(us_holidays_2019.date).astype(int)
    
    # Day before/after holiday (travel surges)
    holiday_adjacent = set()
    for h in us_holidays_2019:
        holiday_adjacent.add((h - pd.Timedelta(days=1)).date())
        holiday_adjacent.add((h + pd.Timedelta(days=1)).date())
    df['is_holiday_adjacent'] = df['time_window'].dt.date.isin(holiday_adjacent).astype(int)
    
    n_added = 15  # 13 base + is_holiday + is_holiday_adjacent
    logger.info(f"Added {n_added} temporal features")
    return df


def add_lag_features(df, config):
    """
    Add lagged demand features per zone.
    
    For each zone, creates features like:
        - lag_1: demand 30 minutes ago
        - lag_2: demand 1 hour ago
        - lag_48: demand 24 hours ago
    
    The data must be sorted by (zone_id, time_window) before calling this.
    """
    lag_windows = config['features']['lag_windows']
    target = config['features']['target']
    
    logger.info(f"Adding lag features: {lag_windows}")
    
    # Ensure sorted
    df = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)
    
    for lag in lag_windows:
        col_name = f'lag_{lag}'
        df[col_name] = df.groupby('zone_id')[target].shift(lag)
    
    logger.info(f"Added {len(lag_windows)} lag features")
    return df


def add_rolling_features(df, config):
    """
    Add rolling window statistics per zone.
    
    For each rolling window size, creates:
        - rolling_mean_{w}: average demand over last w windows
        - rolling_std_{w}: std deviation of demand over last w windows
        - rolling_max_{w}: max demand over last w windows
        - rolling_min_{w}: min demand over last w windows
    
    Uses vectorized shift + rolling to avoid slow lambda transforms.
    """
    rolling_windows = config['features']['rolling_windows']
    target = config['features']['target']
    
    logger.info(f"Adding rolling features for windows: {rolling_windows}")
    
    # Ensure data is sorted by zone and time
    df = df.sort_values(['zone_id', 'time_window']).reset_index(drop=True)
    
    # Shift target by 1 within each zone to prevent data leakage
    df['_shifted_target'] = df.groupby('zone_id')[target].shift(1)
    
    for w in rolling_windows:
        rolled = df.groupby('zone_id')['_shifted_target']
        
        df[f'rolling_mean_{w}'] = rolled.transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        df[f'rolling_std_{w}'] = rolled.transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        ).fillna(0)
        df[f'rolling_max_{w}'] = rolled.transform(
            lambda x: x.rolling(window=w, min_periods=1).max()
        )
        df[f'rolling_min_{w}'] = rolled.transform(
            lambda x: x.rolling(window=w, min_periods=1).min()
        )
    
    # Clean up temporary column
    df = df.drop(columns=['_shifted_target'])
    
    logger.info(f"Added {len(rolling_windows) * 4} rolling features")
    return df


def add_zone_statistics(df, config):
    """
    Add zone-level aggregate statistics as features.
    These capture the baseline demand profile for each zone.
    
    Uses only training data statistics to avoid leakage.
    """
    target = config['features']['target']
    
    # Use first 80% of data for computing zone stats (matches train/test split)
    n_windows = df['time_window'].nunique()
    cutoff_idx = int(n_windows * 0.8)
    sorted_windows = sorted(df['time_window'].unique())
    cutoff_time = sorted_windows[cutoff_idx]
    
    train_data = df[df['time_window'] < cutoff_time]
    
    # Overall zone stats
    zone_stats = train_data.groupby('zone_id')[target].agg(
        zone_mean='mean',
        zone_std='std',
        zone_median='median'
    ).reset_index()
    zone_stats['zone_std'] = zone_stats['zone_std'].fillna(0)
    
    # Zone x hour stats
    zone_hour_stats = train_data.groupby(['zone_id', 'hour'])[target].agg(
        zone_hour_mean='mean'
    ).reset_index()
    
    # Zone x day_of_week stats
    if 'day_of_week' not in train_data.columns:
        train_data = train_data.copy()
        train_data['day_of_week'] = train_data['time_window'].dt.dayofweek
    
    zone_dow_stats = train_data.groupby(['zone_id', 'day_of_week'])[target].agg(
        zone_dow_mean='mean'
    ).reset_index()
    
    # Guard against duplicate columns from repeated merges
    for col in ['zone_mean', 'zone_std', 'zone_median', 'zone_hour_mean', 'zone_dow_mean']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Merge stats
    df = df.merge(zone_stats, on='zone_id', how='left')
    
    if 'hour' in df.columns:
        df = df.merge(zone_hour_stats, on=['zone_id', 'hour'], how='left')
    
    if 'day_of_week' in df.columns:
        df = df.merge(zone_dow_stats, on=['zone_id', 'day_of_week'], how='left')
    
    # Fill any remaining NaN
    for col in ['zone_mean', 'zone_std', 'zone_median', 'zone_hour_mean', 'zone_dow_mean']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    logger.info("Added zone-level statistical features")
    return df


def engineer_features(config, force=False):
    """
    Full feature engineering pipeline.
    
    1. Load aggregated demand data
    2. Add temporal features
    3. Add lag features
    4. Add rolling features
    5. Add zone statistics
    6. Drop rows with NaN from lag/rolling features
    7. Save
    """
    output_path = config['data']['features_file']
    
    if os.path.exists(output_path) and not force:
        logger.info(f"Loading existing features from {output_path}")
        return pd.read_parquet(output_path)
    
    # Load aggregated demand
    agg_path = config['data']['aggregated_file']
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Aggregated data not found at {agg_path}. Run data_ingestion.py first.")
    
    df = pd.read_parquet(agg_path)
    df['time_window'] = pd.to_datetime(df['time_window'])
    logger.info(f"Loaded aggregated data: {df.shape}")
    
    # Feature engineering pipeline
    df = add_temporal_features(df)
    df = add_lag_features(df, config)
    df = add_rolling_features(df, config)
    df = add_zone_statistics(df, config)
    
    # Drop rows where lag/rolling features are NaN (first few time windows per zone)
    max_lag = max(config['features']['lag_windows'])
    max_roll = max(config['features']['rolling_windows'])
    warmup = max(max_lag, max_roll) + 1
    
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {initial_rows - len(df):,} rows with NaN (warmup period)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path} — shape: {df.shape}")
    
    return df


def get_feature_columns(config):
    """Return list of feature column names (excludes target and identifiers)."""
    exclude_cols = ['time_window', 'zone_id', config['features']['target']]
    
    # Load features to get column names
    features_path = config['data']['features_file']
    df = pd.read_parquet(features_path, columns=None)
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    config = load_config()
    
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    df = engineer_features(config)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nFeature columns ({len(df.columns) - 3} features):")
    exclude = ['time_window', 'zone_id', 'trip_count']
    for col in df.columns:
        if col not in exclude:
            print(f"  - {col}")
    
    print(f"\nSample data:")
    print(df.head())
