"""
Visualization Module
====================
Generates publication-quality visualizations for ride demand prediction
failure analysis, including heatmaps, error distributions, temporal
patterns, and spatial failure maps.
"""

import os
import numpy as np
import pandas as pd
import yaml
import logging
import matplotlib
import sys

# Use Agg backend for scripts, but allow notebooks to use their own backend
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1a73e8',
    'failure': '#d93025',
    'success': '#34a853',
    'warning': '#f9ab00',
    'neutral': '#80868b',
    'models': {
        'linear_regression': '#1a73e8',
        'random_forest': '#34a853',
        'lstm': '#f9ab00'
    }
}


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ========================================
# 1. EDA Visualizations
# ========================================
def plot_demand_overview(df, config):
    """Plot overall demand patterns: hourly, daily, monthly distributions."""
    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NYC Taxi Demand Patterns (2019)', fontsize=16, fontweight='bold')
    
    # 1a. Average demand by hour
    hourly = df.groupby('hour')['trip_count'].mean()
    axes[0, 0].bar(hourly.index, hourly.values, color=COLORS['primary'], alpha=0.8)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Avg Trip Count per Zone')
    axes[0, 0].set_title('Average Demand by Hour')
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # 1b. Average demand by day of week
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = df.groupby('day_of_week')['trip_count'].mean()
    axes[0, 1].bar(range(7), daily.values, color=COLORS['primary'], alpha=0.8)
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(dow_names)
    axes[0, 1].set_ylabel('Avg Trip Count per Zone')
    axes[0, 1].set_title('Average Demand by Day of Week')
    
    # 1c. Demand distribution (log scale)
    demand_nonzero = df[df['trip_count'] > 0]['trip_count']
    axes[1, 0].hist(demand_nonzero, bins=100, color=COLORS['primary'], alpha=0.7,
                     edgecolor='white', linewidth=0.5)
    axes[1, 0].set_xlabel('Trip Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Demand Distribution (Non-Zero)')
    axes[1, 0].set_yscale('log')
    
    # 1d. Top 20 zones by average demand
    zone_demand = df.groupby('zone_id')['trip_count'].mean().sort_values(ascending=False).head(20)
    axes[1, 1].barh(range(len(zone_demand)), zone_demand.values, color=COLORS['primary'], alpha=0.8)
    axes[1, 1].set_yticks(range(len(zone_demand)))
    axes[1, 1].set_yticklabels([f'Zone {z}' for z in zone_demand.index])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel('Avg Trip Count per 30-min Window')
    axes[1, 1].set_title('Top 20 Zones by Average Demand')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'demand_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved demand_overview.png")


def plot_demand_heatmap(df, config):
    """Plot heatmap of average demand by hour x day_of_week."""
    figures_dir = config['output']['figures_dir']
    
    pivot = df.groupby(['day_of_week', 'hour'])['trip_count'].mean().reset_index()
    heatmap_data = pivot.pivot(index='day_of_week', columns='hour', values='trip_count')
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, ax=ax,
                xticklabels=range(24), yticklabels=dow_names,
                cbar_kws={'label': 'Avg Trip Count'})
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('')
    ax.set_title('Average Ride Demand Heatmap (Hour × Day of Week)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'demand_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved demand_heatmap.png")


# ========================================
# 2. Model Comparison Visualizations
# ========================================
def plot_model_comparison(metrics_list, config):
    """Bar chart comparing model performance metrics."""
    figures_dir = config['output']['figures_dir']
    
    metrics_df = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    
    for idx, metric in enumerate(['MAE', 'RMSE', 'MAPE']):
        colors = [COLORS['models'].get(m, COLORS['neutral']) for m in metrics_df['model'].str.lower().str.replace(' ', '_', regex=False)]
        axes[idx].bar(metrics_df['model'], metrics_df[metric], color=colors, alpha=0.85)
        axes[idx].set_title(metric, fontsize=12)
        axes[idx].set_ylabel(metric)
        
        # Add value labels
        for i, v in enumerate(metrics_df[metric]):
            axes[idx].text(i, v + 0.01 * v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved model_comparison.png")


def plot_prediction_scatter(errors_dict, config):
    """Scatter plot of true vs predicted values for each model."""
    figures_dir = config['output']['figures_dir']
    
    n_models = len(errors_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('True vs Predicted Demand', fontsize=14, fontweight='bold')
    
    for idx, (model_name, errors_df) in enumerate(errors_dict.items()):
        ax = axes[idx]
        
        # Sample for plotting efficiency
        sample = errors_df.sample(min(10000, len(errors_df)), random_state=42)
        
        color_model = COLORS['models'].get(model_name, COLORS['primary'])
        
        ax.scatter(sample['y_true'], sample['y_pred'], alpha=0.1, s=5, color=color_model)
        
        # Perfect prediction line
        max_val = max(sample['y_true'].max(), sample['y_pred'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=1.5, label='Perfect')
        
        ax.set_xlabel('True Demand')
        ax.set_ylabel('Predicted Demand')
        ax.set_title(model_name.replace('_', ' ').title())
        ax.legend()
        ax.set_xlim(0, min(max_val * 1.1, 500))
        ax.set_ylim(0, min(max_val * 1.1, 500))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'prediction_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved prediction_scatter.png")


# ========================================
# 3. Failure Analysis Visualizations
# ========================================
def plot_failure_rate_by_hour(temporal_analysis, config):
    """Plot failure rate by hour of day for each model."""
    figures_dir = config['output']['figures_dir']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for model_name, analysis in temporal_analysis.items():
        hourly = analysis['hourly']
        color = COLORS['models'].get(model_name, COLORS['neutral'])
        ax.plot(hourly.index, hourly['failure_rate'], marker='o', linewidth=2,
                label=model_name.replace('_', ' ').title(), color=color, markersize=5)
    
    ax.axhline(y=15, color=COLORS['failure'], linestyle='--', alpha=0.5, label='Expected (15%)')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_title('Prediction Failure Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'failure_rate_hourly.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved failure_rate_hourly.png")


def plot_failure_rate_by_day(temporal_analysis, config):
    """Plot failure rate by day of week."""
    figures_dir = config['output']['figures_dir']
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(7)
    width = 0.8 / len(temporal_analysis)
    
    for i, (model_name, analysis) in enumerate(temporal_analysis.items()):
        daily = analysis['daily']
        color = COLORS['models'].get(model_name, COLORS['neutral'])
        ax.bar(x + i * width - 0.4 + width/2, daily['failure_rate'],
               width, label=model_name.replace('_', ' ').title(), color=color, alpha=0.85)
    
    ax.axhline(y=15, color=COLORS['failure'], linestyle='--', alpha=0.5, label='Expected (15%)')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_title('Prediction Failure Rate by Day of Week', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dow_names)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'failure_rate_daily.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved failure_rate_daily.png")


def plot_error_distribution(errors_dict, config):
    """Plot error distribution for each model with failure threshold."""
    figures_dir = config['output']['figures_dir']
    percentile = config['failure_analysis']['error_percentile']
    
    n_models = len(errors_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Prediction Error Distribution', fontsize=14, fontweight='bold')
    
    for idx, (model_name, errors_df) in enumerate(errors_dict.items()):
        ax = axes[idx]
        
        threshold = np.percentile(errors_df['abs_error'], percentile)
        
        ax.hist(errors_df['abs_error'], bins=100, color=COLORS['primary'],
                alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(x=threshold, color=COLORS['failure'], linestyle='--', linewidth=2,
                   label=f'P{percentile} = {threshold:.1f}')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(model_name.replace('_', ' ').title())
        ax.legend(fontsize=10)
        ax.set_xlim(0, np.percentile(errors_df['abs_error'], 99))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved error_distribution.png")


def plot_failure_heatmap(errors_df, model_name, config):
    """
    Heatmap of failure rate by hour x day_of_week for a given model.
    Shows where failures concentrate in the weekly cycle.
    """
    figures_dir = config['output']['figures_dir']
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Compute failure rate per (hour, day_of_week) cell
    pivot = errors_df.groupby(['day_of_week', 'hour']).agg(
        failure_rate=('is_failure', 'mean')
    ).reset_index()
    heatmap_data = pivot.pivot(index='day_of_week', columns='hour', values='failure_rate') * 100
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=False, ax=ax,
                xticklabels=range(24), yticklabels=dow_names,
                cbar_kws={'label': 'Failure Rate (%)'}, vmin=0, vmax=40)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('')
    ax.set_title(f'{model_name.replace("_", " ").title()} - Failure Rate Heatmap (Hour × Day)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'failure_heatmap_{model_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved failure_heatmap_{model_name}.png")


def plot_top_failure_zones(spatial_analysis, config):
    """Bar chart of zones with highest failure rates."""
    figures_dir = config['output']['figures_dir']
    top_n = config['failure_analysis']['top_n_zones']
    
    for model_name, zone_stats in spatial_analysis.items():
        top_zones = zone_stats.head(top_n)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        zone_labels = [f'Zone {z}' for z in top_zones.index]
        if 'Zone' in top_zones.columns:
            zone_labels = [f'{z} ({top_zones.loc[top_zones.index[i], "Zone"]})'
                          for i, z in enumerate(top_zones.index)]
        
        bars = ax.barh(range(len(top_zones)), top_zones['failure_rate'],
                       color=COLORS['failure'], alpha=0.8)
        
        ax.set_yticks(range(len(top_zones)))
        ax.set_yticklabels(zone_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Failure Rate (%)', fontsize=12)
        ax.set_title(f'{model_name.replace("_", " ").title()} - Top {top_n} Zones by Failure Rate',
                     fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, top_zones['failure_rate']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'top_failure_zones_{model_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved top_failure_zones_{model_name}.png")


def plot_demand_vs_error(demand_analysis_dict, config):
    """Plot failure rate by demand level bin for each model."""
    figures_dir = config['output']['figures_dir']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = None
    for model_name, demand_analysis in demand_analysis_dict.items():
        if x_labels is None:
            x_labels = demand_analysis['demand_bin'].astype(str).values
        
        color = COLORS['models'].get(model_name, COLORS['neutral'])
        ax.plot(range(len(demand_analysis)), demand_analysis['failure_rate'],
                marker='s', linewidth=2, markersize=8,
                label=model_name.replace('_', ' ').title(), color=color)
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Demand Level (trips per 30-min window)', fontsize=12)
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_title('Failure Rate by Demand Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'failure_by_demand_level.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved failure_by_demand_level.png")


def plot_monthly_error_trend(errors_dict, config):
    """Plot monthly MAE trend for each model."""
    figures_dir = config['output']['figures_dir']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name, errors_df in errors_dict.items():
        monthly_mae = errors_df.groupby('month')['abs_error'].mean()
        color = COLORS['models'].get(model_name, COLORS['neutral'])
        ax.plot(monthly_mae.index, monthly_mae.values, marker='o', linewidth=2,
                label=model_name.replace('_', ' ').title(), color=color)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Monthly Error Trend', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'monthly_error_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved monthly_error_trend.png")


def generate_all_visualizations(analysis, results, config, demand_df=None):
    """
    Generate all visualization plots.
    
    Args:
        analysis: Output from run_failure_analysis()
        results: Output from run_all_models()
        config: Configuration dict
        demand_df: Optional aggregated demand DataFrame for EDA plots
    """
    logger.info("\nGenerating visualizations...")
    
    # EDA plots (if demand data provided)
    if demand_df is not None:
        if 'hour' not in demand_df.columns:
            demand_df['hour'] = pd.to_datetime(demand_df['time_window']).dt.hour
            demand_df['day_of_week'] = pd.to_datetime(demand_df['time_window']).dt.dayofweek
        plot_demand_overview(demand_df, config)
        plot_demand_heatmap(demand_df, config)
    
    # Model comparison
    plot_model_comparison(results['metrics'], config)
    
    # Prediction scatter
    plot_prediction_scatter(analysis['errors'], config)
    
    # Error distributions
    plot_error_distribution(analysis['errors'], config)
    
    # Failure rate by hour
    plot_failure_rate_by_hour(analysis['temporal'], config)
    
    # Failure rate by day
    plot_failure_rate_by_day(analysis['temporal'], config)
    
    # Failure heatmaps per model
    for model_name, errors_df in analysis['errors'].items():
        plot_failure_heatmap(errors_df, model_name, config)
    
    # Top failure zones
    plot_top_failure_zones(analysis['spatial'], config)
    
    # Failure by demand level
    plot_demand_vs_error(analysis['demand_level'], config)
    
    # Monthly error trend
    plot_monthly_error_trend(analysis['errors'], config)
    
    logger.info("All visualizations generated!")


if __name__ == "__main__":
    config = load_config()
    print("Visualization module ready. Run via notebook or main pipeline.")
