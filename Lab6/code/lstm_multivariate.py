"""
Multivariate LSTM for Traffic Forecasting
Uses all features (Total, CarCount, BikeCount, BusCount, TruckCount) to predict Total
"""
import pandas as pd
import numpy as np
from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
modules_path = os.path.join(project_root, 'package_lab')
if modules_path not in sys.path:
    sys.path.append(modules_path)

# ============================================================================
# DATA PREPARATION FOR MULTIVARIATE LSTM
# ============================================================================

def prepare_multivariate_dataset(data, seq_length: int = 4, target_col_idx: int = 0):
    """
    Transform multivariate time series into sequences for LSTM training
    
    Parameters:
    -----------
    data : numpy array of shape (n_samples, n_features)
    seq_length : int, number of time steps to look back
    target_col_idx : int, index of target column to predict
    
    Returns:
    --------
    X : tensor of shape (n_samples, seq_length, n_features)
    y : tensor of shape (n_samples, 1) - target variable only
    """
    n_samples = len(data) - seq_length
    n_features = data.shape[1]
    
    # Pre-allocate numpy arrays for efficiency
    X = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        # Use all features for past sequences
        X[i] = data[i : i + seq_length, :]  # (seq_length, n_features)
        
        # Predict only the target column at next time step
        y[i] = data[i + seq_length, target_col_idx]  # scalar
    
    return tensor(X), tensor(y)


# ============================================================================
# MULTIVARIATE LSTM MODEL CLASS
# ============================================================================

class MultivariateLSTM(Module):
    def __init__(self, train_data, input_size: int = 5, hidden_size: int = 50, 
                 num_layers: int = 1, seq_length: int = 4, target_col_idx: int = 0,
                 batch_size: int = 32):
        """
        Multivariate LSTM model
        
        Parameters:
        -----------
        train_data : numpy array of shape (n_samples, n_features)
        input_size : int, number of input features
        hidden_size : int, number of hidden units
        num_layers : int, number of LSTM layers
        seq_length : int, sequence length
        target_col_idx : int, index of target column
        batch_size : int, batch size for training
        """
        super().__init__()
        
        self.lstm = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Output layer predicts only target variable
        self.linear = Linear(hidden_size, 1)
        
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss_fn = MSELoss()
        
        # Prepare training data
        trnX, trnY = prepare_multivariate_dataset(train_data, seq_length=seq_length, 
                                                   target_col_idx=target_col_idx)
        
        self.loader = DataLoader(
            TensorDataset(trnX, trnY), 
            shuffle=True, 
            batch_size=min(batch_size, max(1, len(trnX) // 10))
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : tensor of shape (batch_size, seq_length, n_features)
        
        Returns:
        --------
        output : tensor of shape (batch_size, 1)
        """
        # LSTM output: (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last time step's output
        last_time_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Linear layer to predict target
        output = self.linear(last_time_step)  # (batch_size, 1)
        
        return output
    
    def fit(self):
        """Train for one epoch"""
        self.train()
        total_loss = 0
        n_batches = 0
        
        for batchX, batchY in self.loader:
            # Forward pass
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def predict(self, X):
        """Make predictions"""
        self.eval()
        with no_grad():
            y_pred = self(X)
        return y_pred


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate R2, MAE, RMSE"""
    # Convert to numpy arrays
    if hasattr(y_true, 'numpy'):
        y_true_flat = y_true.numpy().flatten()
    elif hasattr(y_true, 'values'):
        y_true_flat = y_true.values.flatten()
    else:
        y_true_flat = np.array(y_true).flatten()
    
    if hasattr(y_pred, 'numpy'):
        y_pred_flat = y_pred.numpy().flatten()
    elif hasattr(y_pred, 'values'):
        y_pred_flat = y_pred.values.flatten()
    else:
        y_pred_flat = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}


# ============================================================================
# HYPERPARAMETER STUDY
# ============================================================================

def lstm_multivariate_study(train_data, test_data, nr_episodes: int = 500, 
                            measure: str = "R2", target_col_idx: int = 0,
                            periodicity: int = 96):
    """
    Study multivariate LSTM performance across different hyperparameters
    
    Parameters:
    -----------
    train_data : numpy array of shape (n_samples, n_features)
    test_data : numpy array of shape (n_samples, n_features)
    nr_episodes : int, number of training epochs
    measure : str, metric to optimize
    target_col_idx : int, index of target column
    periodicity : int, data periodicity (for sequence length selection)
    """
    print(f"\n{'='*60}")
    print("MULTIVARIATE LSTM HYPERPARAMETER STUDY")
    print(f"{'='*60}\n")
    
    input_size = train_data.shape[1]
    print(f"Input features: {input_size}")
    print(f"Target column index: {target_col_idx}")
    
    # Sequence lengths based on periodicity (heavily reduced for speed)
    if periodicity and periodicity > 10:
        sequence_sizes = [periodicity // 2]  # Just one sequence length
        print(f"Using sequence lengths based on periodicity={periodicity}: {sequence_sizes}")
    else:
        sequence_sizes = [16]
    
    hidden_units = [50]  # Just one hidden size
    
    step = max(1, nr_episodes // 10)
    episodes = [1] + list(range(step, nr_episodes + 1, step))
    
    best_model = None
    best_params = {"name": "Multivariate LSTM", "metric": measure, "params": ()}
    best_performance = -100000
    
    fig, axs = plt.subplots(1, len(sequence_sizes), figsize=(15, 5))
    if len(sequence_sizes) == 1:
        axs = [axs]
    
    for i, seq_length in enumerate(sequence_sizes):
        print(f"\n{'='*60}")
        print(f"Testing sequence length: {seq_length}")
        print(f"{'='*60}")
        
        # Prepare test data
        tstX, tstY = prepare_multivariate_dataset(test_data, seq_length=seq_length, 
                                                   target_col_idx=target_col_idx)
        
        values = {}
        
        for hidden in hidden_units:
            print(f"\nHidden units: {hidden}")
            yvalues = []
            
            model = MultivariateLSTM(
                train_data, 
                input_size=input_size,
                hidden_size=hidden, 
                seq_length=seq_length,
                target_col_idx=target_col_idx,
                batch_size=128  # Larger batch size for speed
            )
            
            for n in range(1, nr_episodes + 1):
                loss = model.fit()
                
                if n == 1 or n % step == 0:
                    prd_tst = model.predict(tstX)
                    metrics = calculate_metrics(tstY, prd_tst)
                    eval_score = metrics[measure]
                    
                    print(f"  Episode {n:4d} | Loss: {loss:.4f} | {measure}: {eval_score:.4f}")
                    
                    if eval_score > best_performance and abs(eval_score - best_performance) > 0.01:
                        best_performance = eval_score
                        best_params["params"] = (seq_length, hidden, n)
                        best_model = deepcopy(model)
                    
                    yvalues.append(eval_score)
            
            values[hidden] = yvalues
        
        # Plot results for this sequence length
        ax = axs[i]
        for hidden, vals in values.items():
            ax.plot(episodes, vals, marker='o', label=f'Hidden={hidden}', linewidth=2)
        
        ax.set_title(f'Multivariate LSTM\nseq_length={seq_length}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episodes', fontsize=11)
        ax.set_ylabel(measure, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multivariate LSTM Hyperparameter Study ({measure})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Lab6/results/LSTM_Multi/hyperparameter_study.png', dpi=300)
    
    print(f"\n{'='*60}")
    print(f"BEST RESULTS:")
    print(f"  Sequence length: {best_params['params'][0]}")
    print(f"  Hidden units: {best_params['params'][1]}")
    print(f"  Episodes: {best_params['params'][2]}")
    print(f"  {measure}: {best_performance:.4f}")
    print(f"{'='*60}\n")
    
    return best_model, best_params


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_forecasting_eval(test_true, test_pred, title=""):
    """Plot evaluation metrics"""
    metrics = calculate_metrics(test_true, test_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # R2 Score
    axes[0].bar(['Test'], [metrics['R2']], color='#e74c3c', alpha=0.7, width=0.5)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score (Higher is Better)')
    axes[0].set_ylim([max(-1, metrics['R2'] - 0.2), 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].text(0, metrics['R2'] + 0.05, f"{metrics['R2']:.4f}", 
                ha='center', fontsize=11, fontweight='bold')
    
    # MAE
    axes[1].bar(['Test'], [metrics['MAE']], color='#2ecc71', alpha=0.7, width=0.5)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error (Lower is Better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(0, metrics['MAE'] + metrics['MAE'] * 0.05, f"{metrics['MAE']:.4f}", 
                ha='center', fontsize=11, fontweight='bold')
    
    # RMSE
    axes[2].bar(['Test'], [metrics['RMSE']], color='#3498db', alpha=0.7, width=0.5)
    axes[2].set_ylabel('RMSE', fontsize=12)
    axes[2].set_title('Root Mean Squared Error (Lower is Better)')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].text(0, metrics['RMSE'] + metrics['RMSE'] * 0.05, f"{metrics['RMSE']:.4f}", 
                ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return metrics


def plot_forecasting_series(train_df, test_df, pred_series, title="", target_col="Total"):
    """Plot actual vs predicted time series"""
    plt.figure(figsize=(15, 6))
    
    # Plot training data (last 500 points for context)
    plt.plot(train_df.index[-500:], train_df[target_col].iloc[-500:], 
             label='Train', color='#95a5a6', alpha=0.6, linewidth=1.5)
    
    # Plot actual test data
    plt.plot(test_df.index, test_df[target_col], 
             label='Test (Actual)', color='#2ecc71', linewidth=2)
    
    # Plot predictions
    plt.plot(pred_series.index, pred_series.values, 
             label='Test (Predicted)', color='#e74c3c', 
             linestyle='--', linewidth=2)
    
    # Mark train/test split
    plt.axvline(x=test_df.index[0], color='black', linestyle=':', 
               linewidth=2, label='Train/Test Split', alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel(f'{target_col} Traffic Volume', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("Lab6/results/LSTM_Multi", exist_ok=True)
    
    print("Loading multivariate data...")
    train_df = pd.read_csv('Datasets/traffic_forecasting/processed_train_multi.csv', 
                           parse_dates=['Datetime'])
    test_df = pd.read_csv('Datasets/traffic_forecasting/processed_test_multi.csv', 
                          parse_dates=['Datetime'])
    
    train_df.set_index('Datetime', inplace=True)
    test_df.set_index('Datetime', inplace=True)
    
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Convert to numpy arrays
    train_data = train_df.values.astype('float32')
    test_data = test_df.values.astype('float32')
    
    # Target column is 'Total' (first column, index 0)
    TARGET_COL = 'Total'
    target_col_idx = train_df.columns.tolist().index(TARGET_COL)
    
    print(f"\nTarget column: {TARGET_COL} (index {target_col_idx})")
    
    # Run hyperparameter study
    print("\nStarting multivariate LSTM hyperparameter study...")
    measure = "R2"
    periodicity = 96  # Traffic data has 96 time steps per day (15-min intervals)
    
    best_model, best_params = lstm_multivariate_study(
        train_data, 
        test_data, 
        nr_episodes=500,  # Reduced for speed
        measure=measure, 
        target_col_idx=target_col_idx,
        periodicity=periodicity
    )
    
    # Get best parameters
    best_length, best_hidden, best_episodes = best_params["params"]
    
    # Generate final predictions
    print("\nGenerating final predictions...")
    trnX, trnY = prepare_multivariate_dataset(train_data, seq_length=best_length, 
                                               target_col_idx=target_col_idx)
    tstX, tstY = prepare_multivariate_dataset(test_data, seq_length=best_length, 
                                               target_col_idx=target_col_idx)
    
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    
    # Plot evaluation
    print("\nPlotting evaluation metrics...")
    test_metrics = plot_forecasting_eval(
        tstY, 
        prd_tst,
        title=f"Multivariate LSTM Evaluation\n(seq_len={best_length}, hidden={best_hidden}, epochs={best_episodes})"
    )
    plt.savefig('Lab6/results/LSTM_Multi/evaluation.png', dpi=300)
    print("Saved: Lab6/results/LSTM_Multi/evaluation.png")
    
    # Plot forecast series
    print("Plotting forecast...")
    pred_series = pd.Series(
        prd_tst.numpy().ravel(), 
        index=test_df.index[best_length:]
    )
    
    plot_forecasting_series(
        train_df,
        test_df[best_length:],
        pred_series,
        title=f"Multivariate LSTM Time Series Forecast\n(seq_len={best_length}, hidden={best_hidden})",
        target_col=TARGET_COL
    )
    plt.savefig('Lab6/results/LSTM_Multi/forecast.png', dpi=300)
    print("Saved: Lab6/results/LSTM_Multi/forecast.png")
    
    # Print final metrics
    print("\n" + "="*60)
    print("FINAL METRICS (Test Set):")
    print("="*60)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("MULTIVARIATE LSTM MODELING COMPLETE")
    print("="*60)
    print("\nAll plots saved to Lab6/results/LSTM_Multi/")
