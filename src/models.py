"""
Modeling Module
===============
Implements three prediction models for ride demand:
1. Linear Regression (baseline)
2. Random Forest Regressor (non-linear)
3. LSTM (deep learning, temporal patterns)

All models use a temporal train/test split (last 20% = test).
"""

import os
import numpy as np
import pandas as pd
import yaml
import joblib
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def temporal_train_test_split(df, config):
    """
    Split data temporally — last 20% of time windows form the test set.
    This prevents future data leakage (unlike random splitting).
    """
    test_size = config['models']['test_size']
    
    sorted_windows = sorted(df['time_window'].unique())
    n_windows = len(sorted_windows)
    cutoff_idx = int(n_windows * (1 - test_size))
    cutoff_time = sorted_windows[cutoff_idx]
    
    train = df[df['time_window'] < cutoff_time].copy()
    test = df[df['time_window'] >= cutoff_time].copy()
    
    logger.info(f"Temporal split at {cutoff_time}")
    logger.info(f"  Train: {len(train):,} rows ({train['time_window'].min()} to {train['time_window'].max()})")
    logger.info(f"  Test:  {len(test):,} rows ({test['time_window'].min()} to {test['time_window'].max()})")
    
    return train, test


def get_feature_columns(df, config):
    """Get feature column names."""
    exclude = ['time_window', 'zone_id', config['features']['target']]
    return [c for c in df.columns if c not in exclude]


def prepare_data(train, test, config):
    """Prepare X, y arrays and scale features."""
    target = config['features']['target']
    feature_cols = get_feature_columns(train, config)
    
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[target].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[target].values.astype(np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaler, feature_cols


def evaluate_model(y_true, y_pred, model_name):
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (avoid division by zero, clip extreme values)
    mask = y_true > 0
    if mask.sum() > 0:
        pct_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        # Clip individual percentage errors at 10x (1000%) to avoid outlier distortion
        pct_errors = np.clip(pct_errors, 0, 10.0)
        mape = np.mean(pct_errors) * 100
    else:
        mape = np.nan
    
    metrics = {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    logger.info(f"\n{'='*40}")
    logger.info(f"  {model_name} Results")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"{'='*40}")
    
    return metrics


# ========================================
# Model 1: Linear Regression
# ========================================
def train_linear_regression(X_train, y_train, X_test, y_test, config):
    """Train and evaluate Linear Regression baseline."""
    logger.info("Training Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Demand can't be negative
    
    metrics = evaluate_model(y_test, y_pred, "Linear Regression")
    
    return model, y_pred, metrics


# ========================================
# Model 2: Random Forest
# ========================================
def train_random_forest(X_train, y_train, X_test, y_test, config):
    """Train and evaluate Random Forest Regressor."""
    logger.info("Training Random Forest...")
    
    rf_config = config['models']['random_forest']
    model = RandomForestRegressor(**rf_config)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    
    return model, y_pred, metrics


# ========================================
# Model 3: LSTM
# ========================================
def prepare_lstm_sequences(df, config, scaler=None):
    """
    Prepare sequential data for LSTM.
    
    For LSTM, we create sequences per zone:
    Each sample = (sequence_length time steps of features) -> predict next demand.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    target = config['features']['target']
    feature_cols = get_feature_columns(df, config)
    seq_len = config['models']['lstm']['sequence_length']
    batch_size = config['models']['lstm']['batch_size']
    
    sequences = []
    targets = []
    
    # Create sequences per zone
    zones = sorted(df['zone_id'].unique())
    
    # Sample zones if too many for memory efficiency
    # Use all zones but process efficiently
    for zone_id in zones:
        zone_data = df[df['zone_id'] == zone_id].sort_values('time_window')
        
        X_zone = zone_data[feature_cols].values.astype(np.float32)
        y_zone = zone_data[target].values.astype(np.float32)
        
        if scaler is not None:
            X_zone = scaler.transform(X_zone)
        
        # Create sliding window sequences
        for i in range(seq_len, len(X_zone)):
            sequences.append(X_zone[i - seq_len:i])
            targets.append(y_zone[i])
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, X.shape


class LSTMModel:
    """LSTM wrapper compatible with sklearn-style interface."""
    
    def __init__(self, input_size, config):
        import torch
        import torch.nn as nn
        
        self.config = config['models']['lstm']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._build_model(input_size)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = nn.MSELoss()
    
    def _build_model(self, input_size):
        import torch.nn as nn
        
        class _LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True
                )
                self.fc1 = nn.Linear(hidden_size, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                out = self.fc1(last_hidden)
                out = self.relu(out)
                out = self.fc2(out)
                return out.squeeze(-1)
        
        return _LSTM(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            # PyTorch LSTM requires dropout=0 when num_layers=1
            dropout=self.config['dropout'] if self.config['num_layers'] > 1 else 0.0
        )
    
    def fit(self, train_loader, val_loader=None):
        import torch
        
        epochs = self.config['epochs']
        patience = self.config['patience']
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_state)
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
    
    def _evaluate_loss(self, dataloader):
        import torch
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, dataloader):
        import torch
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())
        
        return np.concatenate(predictions)


def train_lstm(train_df, test_df, config, scaler):
    """Train and evaluate LSTM model.
    
    Uses a validation split from the training data (last 15% of training time)
    for early stopping, to avoid data leakage from the test set.
    """
    logger.info("Training LSTM...")
    
    # Split train into train/val temporally (last 15% of training period for validation)
    train_windows = sorted(train_df['time_window'].unique())
    val_cutoff_idx = int(len(train_windows) * 0.85)
    val_cutoff_time = train_windows[val_cutoff_idx]
    
    train_split = train_df[train_df['time_window'] < val_cutoff_time]
    val_split = train_df[train_df['time_window'] >= val_cutoff_time]
    
    logger.info(f"  LSTM train split: {len(train_split):,} rows")
    logger.info(f"  LSTM val split: {len(val_split):,} rows")
    
    logger.info("Preparing sequences...")
    
    train_loader, train_shape = prepare_lstm_sequences(train_split, config, scaler)
    val_loader, val_shape = prepare_lstm_sequences(val_split, config, scaler)
    test_loader, test_shape = prepare_lstm_sequences(test_df, config, scaler)
    
    logger.info(f"  Train sequences: {train_shape}")
    logger.info(f"  Val sequences: {val_shape}")
    logger.info(f"  Test sequences: {test_shape}")
    
    if train_shape[0] == 0:
        raise ValueError("No training sequences created. Check sequence_length vs data size.")
    
    input_size = train_shape[2]  # number of features
    
    model = LSTMModel(input_size, config)
    model.fit(train_loader, val_loader=val_loader)
    
    y_pred = model.predict(test_loader)
    y_pred = np.maximum(y_pred, 0)
    
    # Get actual y values from test loader
    y_test = []
    for _, y_batch in test_loader:
        y_test.append(y_batch.numpy())
    y_test = np.concatenate(y_test)
    
    metrics = evaluate_model(y_test, y_pred, "LSTM")
    
    return model, y_pred, y_test, metrics


# ========================================
# Main Training Pipeline
# ========================================
def run_all_models(config, sample_zones=None, skip_lstm=False):
    """
    Run the complete modeling pipeline.
    
    Args:
        config: Configuration dictionary
        sample_zones: Optional list of zone IDs to use (for faster execution)
        skip_lstm: If True, skip LSTM training
    
    Returns:
        Dictionary with predictions and metrics for all models
    """
    # Load features
    features_path = config['data']['features_file']
    df = pd.read_parquet(features_path)
    df['time_window'] = pd.to_datetime(df['time_window'])
    
    # Optionally sample zones for faster execution
    if sample_zones is not None:
        df = df[df['zone_id'].isin(sample_zones)].reset_index(drop=True)
        logger.info(f"Using {len(sample_zones)} sampled zones")
    
    # Temporal split
    train, test = temporal_train_test_split(df, config)
    
    # Prepare data
    X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaler, feature_cols = \
        prepare_data(train, test, config)
    
    results = {
        'test_df': test.copy(),
        'feature_cols': feature_cols,
        'scaler': scaler,
        'models': {},
        'predictions': {},
        'metrics': []
    }
    
    # Model 1: Linear Regression
    lr_model, lr_pred, lr_metrics = train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, config
    )
    results['models']['linear_regression'] = lr_model
    results['predictions']['linear_regression'] = lr_pred
    results['metrics'].append(lr_metrics)
    
    # Model 2: Random Forest (uses unscaled data)
    rf_model, rf_pred, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test, config
    )
    results['models']['random_forest'] = rf_model
    results['predictions']['random_forest'] = rf_pred
    results['metrics'].append(rf_metrics)
    
    # Model 3: LSTM
    if skip_lstm:
        logger.info("Skipping LSTM training (--skip-lstm flag).")
    else:
        try:
            lstm_model, lstm_pred, lstm_y_test, lstm_metrics = train_lstm(
                train, test, config, scaler
            )
            results['models']['lstm'] = lstm_model
            results['predictions']['lstm'] = lstm_pred
            results['metrics'].append(lstm_metrics)
            results['lstm_y_test'] = lstm_y_test
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Continuing with other models.")
    
    # Save metrics summary
    metrics_df = pd.DataFrame(results['metrics'])
    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"\n{metrics_df.to_string(index=False)}")
    
    return results


def save_results(results, config):
    """Save models, predictions, and metrics."""
    models_dir = config['output']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    # Save sklearn models
    for name in ['linear_regression', 'random_forest']:
        if name in results['models']:
            path = os.path.join(models_dir, f'{name}.joblib')
            joblib.dump(results['models'][name], path)
            logger.info(f"Saved {name} to {path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(results['scaler'], scaler_path)
    
    # Save LSTM
    if 'lstm' in results['models']:
        import torch
        lstm_path = os.path.join(models_dir, 'lstm_model.pt')
        torch.save(results['models']['lstm'].model.state_dict(), lstm_path)
        logger.info(f"Saved LSTM to {lstm_path}")
    
    # Save predictions
    test_df = results['test_df'].copy()
    for name, preds in results['predictions'].items():
        if name != 'lstm':  # LSTM has different length due to sequencing
            test_df[f'pred_{name}'] = preds
    
    pred_path = os.path.join(config['data']['processed_dir'], 'predictions.parquet')
    test_df.to_parquet(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame(results['metrics'])
    metrics_path = os.path.join(config['output']['reports_dir'], 'model_metrics.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return test_df


if __name__ == "__main__":
    config = load_config()
    
    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)
    
    results = run_all_models(config)
    save_results(results, config)
