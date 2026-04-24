"""
Explainability Module (SHAP)
============================
Uses SHAP values to explain which features contribute most to 
prediction failures. Focuses on comparing feature importance in
normal predictions vs. high-error (failure) cases.
"""

import os
import numpy as np
import pandas as pd
import shap
import yaml
import joblib
import logging
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_shap_values(model, X, feature_names, model_name, max_samples=5000):
    """
    Compute SHAP values for a given model and data.
    
    Uses TreeExplainer for Random Forest and KernelExplainer for Linear models.
    Samples data if too large for efficient computation.
    """
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        logger.info(f"Sampled {max_samples} instances for SHAP computation")
    else:
        X_sample = X
        indices = np.arange(len(X))
    
    if model_name == 'random_forest':
        logger.info("Using TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_name == 'linear_regression':
        logger.info("Using LinearExplainer for Linear Regression...")
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        logger.info("Using KernelExplainer (generic)...")
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)
    
    logger.info(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values, X_sample, indices, explainer


def analyze_failure_shap(shap_values, X_sample, feature_names, is_failure_mask,
                          model_name, config):
    """
    Compare SHAP feature importance between normal and failure cases.
    
    Returns:
        DataFrame comparing mean |SHAP| values for normal vs failure cases
    """
    abs_shap = np.abs(shap_values)
    
    # Split into normal and failure groups
    normal_mask = ~is_failure_mask
    failure_mask = is_failure_mask
    
    if failure_mask.sum() == 0:
        logger.warning("No failure cases in sample!")
        return None
    
    normal_importance = pd.DataFrame(
        abs_shap[normal_mask].mean(axis=0).reshape(1, -1),
        columns=feature_names
    ).T.rename(columns={0: 'normal_mean_shap'})
    
    failure_importance = pd.DataFrame(
        abs_shap[failure_mask].mean(axis=0).reshape(1, -1),
        columns=feature_names
    ).T.rename(columns={0: 'failure_mean_shap'})
    
    comparison = normal_importance.join(failure_importance)
    comparison['importance_ratio'] = (
        comparison['failure_mean_shap'] / comparison['normal_mean_shap'].clip(lower=1e-10)
    )
    comparison = comparison.sort_values('failure_mean_shap', ascending=False)
    
    logger.info(f"\n{model_name} - SHAP Feature Importance (Failures vs Normal):")
    logger.info(f"  Normal cases: {normal_mask.sum()}")
    logger.info(f"  Failure cases: {failure_mask.sum()}")
    logger.info(f"\n  Top features by failure importance:")
    logger.info(comparison.head(15).to_string())
    
    return comparison


def generate_shap_plots(shap_values, X_sample, feature_names, is_failure_mask,
                         model_name, config):
    """
    Generate SHAP visualization plots.
    
    Creates:
        1. Overall SHAP summary plot
        2. SHAP summary for failure cases only
        3. Feature importance comparison bar chart
        4. SHAP dependence plots for top features
    """
    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    
    # 1. Overall SHAP Summary (beeswarm)
    # Note: shap.summary_plot creates its own figure internally
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.title(f'{model_name} - SHAP Summary (All Predictions)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'shap_summary_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 2. SHAP Summary for Failures Only
    if is_failure_mask.sum() > 10:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[is_failure_mask], 
            X_df.iloc[np.where(is_failure_mask)[0]], 
            show=False, max_display=20
        )
        plt.title(f'{model_name} - SHAP Summary (Failure Cases Only)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'shap_summary_failures_{model_name}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # 3. Feature Importance Comparison Bar Chart
    abs_shap = np.abs(shap_values)
    normal_importance = abs_shap[~is_failure_mask].mean(axis=0)
    failure_importance = abs_shap[is_failure_mask].mean(axis=0) if is_failure_mask.sum() > 0 else np.zeros_like(normal_importance)
    
    top_k = 15
    top_indices = np.argsort(failure_importance)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(top_k)
    width = 0.35
    
    ax.barh(x + width/2, normal_importance[top_indices], width, label='Normal', color='#2196F3', alpha=0.8)
    ax.barh(x - width/2, failure_importance[top_indices], width, label='Failure', color='#F44336', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'{model_name} - Feature Importance: Normal vs Failure Cases', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'shap_comparison_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. SHAP Dependence Plots for Top 4 Features
    top_4 = np.argsort(failure_importance)[-4:][::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (feat_idx, ax) in enumerate(zip(top_4, axes.flatten())):
        shap.dependence_plot(
            feat_idx, shap_values, X_df,
            ax=ax, show=False,
            interaction_index=None
        )
        ax.set_title(f'{feature_names[feat_idx]}', fontsize=11)
    
    fig.suptitle(f'{model_name} - SHAP Dependence (Top 4 Failure Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'shap_dependence_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP plots for {model_name} to {figures_dir}")


def run_explainability(results, failure_analysis, config):
    """
    Run SHAP explainability analysis for all sklearn models.
    
    Args:
        results: Output from run_all_models()
        failure_analysis: Output from run_failure_analysis()
        config: Configuration dict
    
    Returns:
        Dictionary of SHAP analysis results per model
    """
    test_df = results['test_df']
    feature_cols = results['feature_cols']
    scaler = results['scaler']
    
    shap_results = {}
    
    for model_name in ['random_forest', 'linear_regression']:
        if model_name not in results['models']:
            continue
        
        model = results['models'][model_name]
        errors_df = failure_analysis['errors'].get(model_name)
        
        if errors_df is None:
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SHAP Analysis: {model_name}")
        logger.info(f"{'='*60}")
        
        # Prepare data
        if model_name == 'linear_regression':
            X = scaler.transform(test_df[feature_cols].values.astype(np.float32))
        else:
            X = test_df[feature_cols].values.astype(np.float32)
        
        # Compute SHAP
        shap_values, X_sample, indices, explainer = compute_shap_values(
            model, X, feature_cols, model_name
        )
        
        # Get failure mask for sampled data
        is_failure = errors_df.iloc[indices]['is_failure'].values.astype(bool)
        
        # Analyze
        comparison = analyze_failure_shap(
            shap_values, X_sample, feature_cols, is_failure,
            model_name, config
        )
        
        # Generate plots
        generate_shap_plots(
            shap_values, X_sample, feature_cols, is_failure,
            model_name, config
        )
        
        shap_results[model_name] = {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'indices': indices,
            'is_failure': is_failure,
            'comparison': comparison,
            'explainer': explainer
        }
    
    return shap_results


if __name__ == "__main__":
    config = load_config()
    print("Explainability module ready. Run via notebook or main pipeline.")
