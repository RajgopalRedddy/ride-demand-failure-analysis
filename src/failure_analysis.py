"""
Failure Pattern Analysis Module
================================
Identifies prediction failures (top 10-15% error cases) and analyzes
spatial and temporal patterns in these failures.
"""

import os
import numpy as np
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_errors(test_df, predictions, model_name, config):
    """
    Compute error metrics for each prediction and identify failure cases.
    
    Failure is defined as predictions where the absolute error exceeds
    the 85th percentile (top 15% worst predictions).
    """
    target = config['features']['target']
    percentile = config['failure_analysis']['error_percentile']
    
    y_true = test_df[target].values
    y_pred = predictions
    
    # Compute errors
    errors = pd.DataFrame({
        'time_window': test_df['time_window'].values,
        'zone_id': test_df['zone_id'].values,
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'error': y_true - y_pred,  # Signed error (positive = underprediction)
        'squared_error': (y_true - y_pred) ** 2,
    })
    
    # Percentage error (where true > 0)
    mask = errors['y_true'] > 0
    errors['pct_error'] = np.nan
    errors.loc[mask, 'pct_error'] = (
        np.abs(errors.loc[mask, 'error']) / errors.loc[mask, 'y_true'] * 100
    )
    
    # Add temporal features for analysis
    errors['hour'] = pd.to_datetime(errors['time_window']).dt.hour
    errors['day_of_week'] = pd.to_datetime(errors['time_window']).dt.dayofweek
    errors['is_weekend'] = (errors['day_of_week'] >= 5).astype(int)
    errors['month'] = pd.to_datetime(errors['time_window']).dt.month
    
    # Flag failures
    error_threshold = np.percentile(errors['abs_error'], percentile)
    errors['is_failure'] = (errors['abs_error'] >= error_threshold).astype(int)
    
    errors['model'] = model_name
    
    n_failures = errors['is_failure'].sum()
    logger.info(f"\n{model_name} Error Analysis:")
    logger.info(f"  Total predictions: {len(errors):,}")
    logger.info(f"  Error threshold (P{percentile}): {error_threshold:.2f}")
    logger.info(f"  Failure cases: {n_failures:,} ({n_failures/len(errors)*100:.1f}%)")
    logger.info(f"  Mean abs error (all): {errors['abs_error'].mean():.4f}")
    logger.info(f"  Mean abs error (failures): {errors[errors['is_failure']==1]['abs_error'].mean():.4f}")
    
    return errors


def analyze_temporal_failures(errors_df, model_name):
    """
    Analyze failure patterns across time dimensions.
    
    Returns:
        Dictionary of temporal failure analysis DataFrames
    """
    failures = errors_df[errors_df['is_failure'] == 1]
    all_data = errors_df
    
    results = {}
    
    # 1. Failure rate by hour of day
    hour_total = all_data.groupby('hour').size()
    hour_failures = failures.groupby('hour').size()
    hourly = pd.DataFrame({
        'total_predictions': hour_total,
        'failures': hour_failures,
        'failure_rate': (hour_failures / hour_total * 100).round(2),
        'mean_abs_error': all_data.groupby('hour')['abs_error'].mean(),
        'mean_abs_error_failures': failures.groupby('hour')['abs_error'].mean()
    }).fillna(0)
    results['hourly'] = hourly
    
    # 2. Failure rate by day of week
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    dow_total = all_data.groupby('day_of_week').size()
    dow_failures = failures.groupby('day_of_week').size()
    daily = pd.DataFrame({
        'day_name': [dow_names.get(d, d) for d in range(7)],
        'total_predictions': dow_total,
        'failures': dow_failures,
        'failure_rate': (dow_failures / dow_total * 100).round(2),
        'mean_abs_error': all_data.groupby('day_of_week')['abs_error'].mean()
    }).fillna(0)
    results['daily'] = daily
    
    # 3. Failure rate: weekday vs weekend
    wknd_total = all_data.groupby('is_weekend').size()
    wknd_failures = failures.groupby('is_weekend').size()
    weekend = pd.DataFrame({
        'type': ['Weekday', 'Weekend'],
        'total_predictions': wknd_total.values,
        'failures': wknd_failures.reindex([0, 1], fill_value=0).values,
        'failure_rate': (wknd_failures.reindex([0, 1], fill_value=0) / wknd_total * 100).round(2).values
    })
    results['weekend'] = weekend
    
    # 4. Failure rate by month
    month_total = all_data.groupby('month').size()
    month_failures = failures.groupby('month').size()
    monthly = pd.DataFrame({
        'total_predictions': month_total,
        'failures': month_failures,
        'failure_rate': (month_failures / month_total * 100).round(2),
        'mean_abs_error': all_data.groupby('month')['abs_error'].mean()
    }).fillna(0)
    results['monthly'] = monthly
    
    # 5. Underprediction vs Overprediction in failures
    under = failures[failures['error'] > 0]  # true > pred
    over = failures[failures['error'] < 0]   # true < pred
    results['direction'] = {
        'underpredictions': len(under),
        'overpredictions': len(over),
        'under_pct': len(under) / len(failures) * 100 if len(failures) > 0 else 0,
        'mean_under_error': under['abs_error'].mean() if len(under) > 0 else 0,
        'mean_over_error': over['abs_error'].mean() if len(over) > 0 else 0
    }
    
    logger.info(f"\n{model_name} - Temporal Failure Patterns:")
    logger.info(f"  Peak failure hour: {hourly['failure_rate'].idxmax()}:00 ({hourly['failure_rate'].max():.1f}%)")
    logger.info(f"  Worst day: {daily.loc[daily['failure_rate'].idxmax(), 'day_name']} ({daily['failure_rate'].max():.1f}%)")
    logger.info(f"  Underpredictions: {results['direction']['under_pct']:.1f}%")
    
    return results


def analyze_spatial_failures(errors_df, model_name, config, zone_lookup=None):
    """
    Analyze failure patterns across taxi zones.
    
    Returns:
        DataFrame with per-zone failure statistics
    """
    top_n = config['failure_analysis']['top_n_zones']
    failures = errors_df[errors_df['is_failure'] == 1]
    all_data = errors_df
    
    # Per-zone statistics
    zone_total = all_data.groupby('zone_id').size()
    zone_failures = failures.groupby('zone_id').size()
    zone_mae = all_data.groupby('zone_id')['abs_error'].mean()
    zone_mean_demand = all_data.groupby('zone_id')['y_true'].mean()
    
    zone_stats = pd.DataFrame({
        'total_predictions': zone_total,
        'failures': zone_failures,
        'failure_rate': (zone_failures / zone_total * 100).round(2),
        'mean_abs_error': zone_mae.round(4),
        'mean_demand': zone_mean_demand.round(2),
        'max_error': all_data.groupby('zone_id')['abs_error'].max().round(2)
    }).fillna(0)
    
    zone_stats = zone_stats.sort_values('failure_rate', ascending=False)
    
    # Add zone names if available
    if zone_lookup is not None:
        zone_stats = zone_stats.merge(
            zone_lookup[['LocationID', 'Zone', 'Borough']],
            left_index=True, right_on='LocationID', how='left'
        ).set_index('LocationID')
    
    logger.info(f"\n{model_name} - Top {top_n} Failure Zones:")
    logger.info(zone_stats.head(top_n).to_string())
    
    return zone_stats


def analyze_demand_level_failures(errors_df, model_name):
    """
    Analyze how failures correlate with demand levels.
    
    Groups predictions into demand bins and examines failure rates.
    """
    df = errors_df.copy()
    
    # Create demand bins
    bins = [0, 1, 5, 10, 25, 50, 100, 250, 500, float('inf')]
    labels = ['0', '1-4', '5-9', '10-24', '25-49', '50-99', '100-249', '250-499', '500+']
    
    df['demand_bin'] = pd.cut(df['y_true'], bins=bins, labels=labels, right=False)
    
    demand_analysis = df.groupby('demand_bin', observed=True).agg(
        total=('is_failure', 'count'),
        failures=('is_failure', 'sum'),
        mean_abs_error=('abs_error', 'mean'),
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean')
    ).reset_index()
    
    demand_analysis['failure_rate'] = (demand_analysis['failures'] / demand_analysis['total'] * 100).round(2)
    
    logger.info(f"\n{model_name} - Failure Rate by Demand Level:")
    logger.info(demand_analysis.to_string(index=False))
    
    return demand_analysis


def cross_model_failure_comparison(all_errors, config):
    """
    Compare failure patterns across models.
    Identifies zones/times where ALL models fail vs model-specific failures.
    """
    models = all_errors['model'].unique()
    
    if len(models) < 2:
        logger.info("Need at least 2 models for cross-model comparison")
        return None
    
    # Find common failure points (zone_id, time_window pairs)
    failure_sets = {}
    for model in models:
        model_errors = all_errors[
            (all_errors['model'] == model) & (all_errors['is_failure'] == 1)
        ]
        failure_sets[model] = set(
            zip(model_errors['zone_id'], model_errors['time_window'])
        )
    
    # Common failures across all models
    common_failures = failure_sets[models[0]]
    for model in models[1:]:
        common_failures = common_failures.intersection(failure_sets[model])
    
    total_unique_failures = set()
    for fset in failure_sets.values():
        total_unique_failures = total_unique_failures.union(fset)
    
    logger.info(f"\nCross-Model Failure Analysis:")
    logger.info(f"  Models compared: {list(models)}")
    for model in models:
        logger.info(f"  {model} failures: {len(failure_sets[model]):,}")
    logger.info(f"  Common failures (all models): {len(common_failures):,}")
    logger.info(f"  Total unique failure points: {len(total_unique_failures):,}")
    if len(total_unique_failures) > 0:
        logger.info(f"  Overlap rate: {len(common_failures)/len(total_unique_failures)*100:.1f}%")
    else:
        logger.info(f"  Overlap rate: N/A (no failures found)")
    
    return {
        'failure_sets': failure_sets,
        'common_failures': common_failures,
        'total_unique': total_unique_failures
    }


def run_failure_analysis(results, config, zone_lookup=None):
    """
    Run complete failure analysis pipeline for all models.
    
    Args:
        results: Output from run_all_models()
        config: Configuration dict
        zone_lookup: Optional zone name lookup DataFrame
    
    Returns:
        Dictionary with all analysis results
    """
    test_df = results['test_df']
    analysis = {
        'errors': {},
        'temporal': {},
        'spatial': {},
        'demand_level': {},
    }
    
    all_errors_list = []
    
    for model_name, predictions in results['predictions'].items():
        if model_name == 'lstm':
            # LSTM predictions have different length due to sequence windowing.
            # Use the separately stored lstm_y_test for proper alignment.
            if 'lstm_y_test' not in results:
                logger.warning("LSTM y_test not found, skipping LSTM failure analysis.")
                continue
            
            lstm_y_test = results['lstm_y_test']
            lstm_preds = predictions
            
            # Create a minimal DataFrame for LSTM error analysis
            # Note: we don't have exact zone/time alignment for LSTM sequences,
            # so we compute aggregate metrics only
            lstm_errors = pd.DataFrame({
                'time_window': pd.NaT,  # Not aligned to specific windows
                'zone_id': 0,
                'y_true': lstm_y_test,
                'y_pred': lstm_preds,
                'abs_error': np.abs(lstm_y_test - lstm_preds),
                'error': lstm_y_test - lstm_preds,
                'squared_error': (lstm_y_test - lstm_preds) ** 2,
            })
            lstm_errors['pct_error'] = np.nan
            mask = lstm_errors['y_true'] > 0
            lstm_errors.loc[mask, 'pct_error'] = (
                np.abs(lstm_errors.loc[mask, 'error']) / lstm_errors.loc[mask, 'y_true'] * 100
            )
            
            percentile = config['failure_analysis']['error_percentile']
            threshold = np.percentile(lstm_errors['abs_error'], percentile)
            lstm_errors['is_failure'] = (lstm_errors['abs_error'] >= threshold).astype(int)
            lstm_errors['model'] = model_name
            lstm_errors['hour'] = 0
            lstm_errors['day_of_week'] = 0
            lstm_errors['is_weekend'] = 0
            lstm_errors['month'] = 0
            
            n_failures = lstm_errors['is_failure'].sum()
            logger.info(f"\nLSTM Error Analysis (aggregate only — no zone/time alignment):")
            logger.info(f"  Total predictions: {len(lstm_errors):,}")
            logger.info(f"  Error threshold (P{percentile}): {threshold:.2f}")
            logger.info(f"  Failure cases: {n_failures:,} ({n_failures/len(lstm_errors)*100:.1f}%)")
            logger.info(f"  Mean abs error (all): {lstm_errors['abs_error'].mean():.4f}")
            
            # Store but don't run spatial/temporal analysis (no alignment info)
            analysis['errors'][model_name] = lstm_errors
            continue
        
        # Compute errors and flag failures
        errors = compute_errors(test_df, predictions, model_name, config)
        analysis['errors'][model_name] = errors
        all_errors_list.append(errors)
        
        # Temporal analysis
        analysis['temporal'][model_name] = analyze_temporal_failures(errors, model_name)
        
        # Spatial analysis
        analysis['spatial'][model_name] = analyze_spatial_failures(
            errors, model_name, config, zone_lookup
        )
        
        # Demand level analysis
        analysis['demand_level'][model_name] = analyze_demand_level_failures(errors, model_name)
    
    # Cross-model comparison
    if all_errors_list:
        all_errors = pd.concat(all_errors_list, ignore_index=True)
        analysis['cross_model'] = cross_model_failure_comparison(all_errors, config)
        analysis['all_errors'] = all_errors
    
    return analysis


def save_failure_analysis(analysis, config):
    """Save failure analysis results."""
    reports_dir = config['output']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    
    for model_name, errors_df in analysis['errors'].items():
        path = os.path.join(reports_dir, f'errors_{model_name}.parquet')
        errors_df.to_parquet(path, index=False)
        logger.info(f"Saved errors for {model_name} to {path}")
    
    for model_name, spatial in analysis['spatial'].items():
        path = os.path.join(reports_dir, f'zone_failures_{model_name}.csv')
        spatial.to_csv(path)
        logger.info(f"Saved zone failure analysis for {model_name} to {path}")


if __name__ == "__main__":
    config = load_config()
    print("Failure analysis module ready. Run via notebook or main pipeline.")
