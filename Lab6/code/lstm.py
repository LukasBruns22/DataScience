import pandas as pd
import numpy as np
from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_dataset_for_lstm(series, seq_length: int = 4):
    """Transform time series into sequences for LSTM training"""
    setX = []
    setY = []
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length]
        future = series[i + 1 : i + seq_length + 1]
        setX.append(past)
        setY.append(future)
    return tensor(setX), tensor(setY)


# ============================================================================
# LSTM MODEL CLASS
# ============================================================================

class DS_LSTM(Module):
    def __init__(self, train, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 1, length: int = 4):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = MSELoss()
        
        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        self.loader = DataLoader(
            TensorDataset(trnX, trnY), 
            shuffle=True, 
            batch_size=max(1, len(train) // 10)
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
    def fit(self):
        self.train()
        total_loss = 0
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)
    
    def predict(self, X):
        self.eval()
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate R2, MAE, RMSE, and MAPE"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.numpy().flatten()
    
    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    
    # MAPE with handling for zero values
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ============================================================================
# LSTM HYPERPARAMETER STUDY
# ============================================================================

def lstm_study(train, test, nr_episodes: int = 1000, measure: str = "R2", periodicity: int = None):
    """Study LSTM performance across different hyperparameters"""
    # For periodic data (like traffic with periodicity=96), use longer sequences
    if periodicity and periodicity > 10:
        sequence_sizes = [periodicity // 2, periodicity, periodicity * 2]
        print(f"Detected periodicity={periodicity}. Using sequence lengths: {sequence_sizes}")
    else:
        sequence_sizes = [2, 4, 8]  # Default for non-periodic data
    
    nr_hidden_units = [25, 50, 100]
    step = nr_episodes // 10
    episodes = [1] + list(range(step, nr_episodes + 1, step))
    
    best_model = None
    best_params = {"name": "LSTM", "metric": measure, "params": ()}
    best_performance = -100000
    
    fig, axs = plt.subplots(1, len(sequence_sizes), figsize=(15, 5))
    
    for i, length in enumerate(sequence_sizes):
        print(f"\n{'='*60}")
        print(f"Testing sequence length: {length}")
        print(f"{'='*60}")
        
        tstX, tstY = prepare_dataset_for_lstm(test, seq_length=length)
        values = {}
        
        for hidden in nr_hidden_units:
            print(f"\nHidden units: {hidden}")
            yvalues = []
            model = DS_LSTM(train, hidden_size=hidden, length=length)
            
            for n in range(1, nr_episodes + 1):
                loss = model.fit()
                
                if n == 1 or n % step == 0:
                    prd_tst = model.predict(tstX)
                    metrics = calculate_metrics(test[length:], prd_tst)
                    eval_score = metrics[measure]
                    
                    print(f"  Episode {n:4d} | Loss: {loss:.4f} | {measure}: {eval_score:.4f}")
                    
                    if eval_score > best_performance and abs(eval_score - best_performance) > 0.01:
                        best_performance = eval_score
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)
                    
                    yvalues.append(eval_score)
            
            values[hidden] = yvalues
        
        # Plot results for this sequence length
        ax = axs[i] if len(sequence_sizes) > 1 else axs
        for hidden, vals in values.items():
            ax.plot(episodes, vals, marker='o', label=f'Hidden={hidden}')
        
        ax.set_title(f'LSTM seq_length={length} ({measure})')
        ax.set_xlabel('Episodes')
        ax.set_ylabel(measure)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_hyperparameter_study.png', dpi=300, bbox_inches='tight')
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

def plot_forecasting_eval(train_true, test_true, train_pred, test_pred, title=""):
    """Plot evaluation metrics comparing train and test performance"""
    train_metrics = calculate_metrics(train_true, train_pred)
    test_metrics = calculate_metrics(test_true, test_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # R2 Score
    axes[0, 0].bar(['Train', 'Test'], [train_metrics['R2'], test_metrics['R2']], color=['#3498db', '#e74c3c'])
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_ylim([min(0, min(train_metrics['R2'], test_metrics['R2']) - 0.1), 1])
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].bar(['Train', 'Test'], [train_metrics['MAE'], test_metrics['MAE']], color=['#3498db', '#e74c3c'])
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].bar(['Train', 'Test'], [train_metrics['RMSE'], test_metrics['RMSE']], color=['#3498db', '#e74c3c'])
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE
    axes[1, 1].bar(['Train', 'Test'], [train_metrics['MAPE'], test_metrics['MAPE']], color=['#3498db', '#e74c3c'])
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_evaluation.png', dpi=300, bbox_inches='tight')


def plot_forecasting_series(train, test, pred_series, title="", xlabel="", ylabel=""):
    """Plot actual vs predicted time series"""
    plt.figure(figsize=(15, 6))
    
    plt.plot(train.index, train.values, label='Train', color='#3498db', alpha=0.7)
    plt.plot(test.index, test.values, label='Test (Actual)', color='#2ecc71', alpha=0.7)
    plt.plot(pred_series.index, pred_series.values, label='Test (Predicted)', 
             color='#e74c3c', linestyle='--', linewidth=2)
    
    plt.axvline(x=test.index[0], color='black', linestyle=':', linewidth=2, label='Train/Test Split')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_forecast.png', dpi=300, bbox_inches='tight')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('datasets/traffic_forecasting/processed_train.csv', parse_dates=['Datetime'])
    test_df = pd.read_csv('datasets/traffic_forecasting/processed_test.csv', parse_dates=['Datetime'])
    
    train_df.set_index('Datetime', inplace=True)
    test_df.set_index('Datetime', inplace=True)
    
    # Convert to numpy arrays for LSTM
    train = train_df[['Total']].values.astype('float32')
    test = test_df[['Total']].values.astype('float32')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Run hyperparameter study
    print("\nStarting LSTM hyperparameter study...")
    measure = "R2"
    periodicity = 96  # Traffic data has 96 time steps per day (15-min intervals)
    best_model, best_params = lstm_study(train, test, nr_episodes=500, measure=measure, periodicity=periodicity)
    
    # Get best parameters
    best_length, best_hidden, best_episodes = best_params["params"]

    # Generate predictions
    print("\nGenerating final predictions...")
    trnX, trnY = prepare_dataset_for_lstm(train, seq_length=best_length)
    tstX, tstY = prepare_dataset_for_lstm(test, seq_length=best_length)
    
    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    
    # Plot evaluation
    plot_forecasting_eval(
        train[best_length:], 
        test[best_length:], 
        prd_trn, 
        prd_tst,
        title=f"LSTM Evaluation (length={best_length}, hidden={best_hidden}, epochs={best_episodes})"
    )
    
    # Plot forecast series
    pred_series = pd.Series(
        prd_tst.numpy().ravel(), 
        index=test_df.index[best_length:]
    )
    
    plot_forecasting_series(
        train_df['Total'][best_length:],
        test_df['Total'][best_length:],
        pred_series,
        title="LSTM Time Series Forecast",
        xlabel="Datetime",
        ylabel="Total"
    )
    
    # Print final metrics
    print("\n" + "="*60)
    print("FINAL METRICS:")
    print("="*60)
    train_metrics = calculate_metrics(train[best_length:], prd_trn)
    test_metrics = calculate_metrics(test[best_length:], prd_tst)
    
    print("\nTraining Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nPlots saved:")
    print("  - lstm_hyperparameter_study.png")
    print("  - lstm_evaluation.png")
    print("  - lstm_forecast.png")