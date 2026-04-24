"""
Main Pipeline
=============
End-to-end execution of the ride demand failure analysis project.

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-download    # Skip data download (if data already exists)
    python main.py --sample-zones 50  # Use only 50 zones (faster for testing)

Steps:
    1. Data Download & Aggregation
    2. Feature Engineering
    3. Model Training (Linear Regression, Random Forest, LSTM)
    4. Failure Pattern Analysis
    5. SHAP Explainability
    6. Visualization Generation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import load_config, download_taxi_data, download_zone_lookup, aggregate_demand
from src.feature_engineering import engineer_features
from src.models import run_all_models, save_results
from src.failure_analysis import run_failure_analysis, save_failure_analysis
from src.explainability import run_explainability
from src.visualizations import generate_all_visualizations

project_root = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'pipeline.log'))
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Ride Demand Failure Analysis Pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--sample-zones', type=int, default=None,
                        help='Number of zones to sample (for faster execution)')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM training (faster)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    start_time = time.time()
    
    # ========================================
    # STEP 1: Data Download & Aggregation
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 1: DATA DOWNLOAD & AGGREGATION")
    print("=" * 70)
    
    if not args.skip_download:
        download_taxi_data(config)
    else:
        logger.info("Skipping data download.")
    
    zone_lookup = download_zone_lookup(config)
    demand_df = aggregate_demand(config)
    
    logger.info(f"Aggregated data: {demand_df.shape}")
    logger.info(f"  Time range: {demand_df['time_window'].min()} -> {demand_df['time_window'].max()}")
    logger.info(f"  Zones: {demand_df['zone_id'].nunique()}")
    logger.info(f"  Total trips: {demand_df['trip_count'].sum():,}")
    
    # ========================================
    # STEP 2: Feature Engineering
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    features_df = engineer_features(config)
    logger.info(f"Engineered features: {features_df.shape}")
    
    # ========================================
    # STEP 3: Model Training
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 3: MODEL TRAINING")
    print("=" * 70)
    
    sample_zones = None
    if args.sample_zones:
        all_zones = sorted(features_df['zone_id'].unique())
        np.random.seed(42)
        sample_zones = sorted(np.random.choice(all_zones, args.sample_zones, replace=False))
        logger.info(f"Sampling {args.sample_zones} zones for modeling")
    
    results = run_all_models(config, sample_zones=sample_zones, skip_lstm=args.skip_lstm)
    test_df = save_results(results, config)
    
    # ========================================
    # STEP 4: Failure Pattern Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 4: FAILURE PATTERN ANALYSIS")
    print("=" * 70)
    
    analysis = run_failure_analysis(results, config, zone_lookup)
    save_failure_analysis(analysis, config)
    
    # ========================================
    # STEP 5: SHAP Explainability
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 5: SHAP EXPLAINABILITY")
    print("=" * 70)
    
    try:
        shap_results = run_explainability(results, analysis, config)
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        shap_results = None
    
    # ========================================
    # STEP 6: Visualization
    # ========================================
    print("\n" + "=" * 70)
    print("  STEP 6: VISUALIZATION")
    print("=" * 70)
    
    generate_all_visualizations(analysis, results, config, demand_df)
    
    # ========================================
    # Summary
    # ========================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total execution time: {elapsed/60:.1f} minutes")
    print(f"\n  Outputs:")
    print(f"    Models:   {config['output']['models_dir']}/")
    print(f"    Figures:  {config['output']['figures_dir']}/")
    print(f"    Reports:  {config['output']['reports_dir']}/")
    
    if results['metrics']:
        print(f"\n  Model Performance Summary:")
        metrics_df = pd.DataFrame(results['metrics'])
        print(f"    {metrics_df.to_string(index=False)}")
    
    print("\n  Done!")


if __name__ == "__main__":
    main()
