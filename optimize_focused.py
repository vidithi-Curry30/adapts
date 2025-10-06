import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts import AdapTS
import matplotlib.pyplot as plt
import json

print("="*70)
print("AdapTS Focused Optimization: 5 Train + 7 Test Datasets")
print("="*70)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

TRAINING_TICKERS = ['SPY', 'NVDA', 'GLD', 'XOM', 'JNJ']  # 5 diverse for training
TESTING_TICKERS = ['GOOGL', 'MSFT', 'WMT', 'COST', 'USO', 'CVX', 'UNH']  # 7 for testing

# ============================================================================
# STEP 1: LOAD TRAINING DATASETS
# ============================================================================

print("\n[1] Loading TRAINING datasets...")
train_datasets = {}

for ticker in TRAINING_TICKERS:
    try:
        loader = TimeSeriesDataLoader(source='stock')
        train, test = loader.load(ticker=ticker, start='2015-01-01', end='2024-01-01')
        train_datasets[ticker] = {'train': train, 'test': test}
        print(f"  ✓ {ticker}: {len(train)} train samples")
    except Exception as e:
        print(f"  ✗ {ticker}: Failed - {e}")

# ============================================================================
# STEP 2: LOAD TESTING DATASETS
# ============================================================================

print("\n[2] Loading TESTING datasets...")
test_datasets = {}

for ticker in TESTING_TICKERS:
    try:
        loader = TimeSeriesDataLoader(source='stock')
        train, test = loader.load(ticker=ticker, start='2015-01-01', end='2024-01-01')
        test_datasets[ticker] = {'train': train, 'test': test}
        print(f"  ✓ {ticker}: {len(test)} test samples")
    except Exception as e:
        print(f"  ✗ {ticker}: Failed - {e}")

# ============================================================================
# STEP 3: TRAIN FOUNDATION MODEL
# ============================================================================

print("\n[3] Training Foundation Model on 5 training datasets...")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=24):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y

# Combine training data
all_train_data = np.concatenate([d['train'] for d in train_datasets.values()])
train_dataset = TimeSeriesDataset(all_train_data, seq_len=96, pred_len=24)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"  Combined training samples: {len(train_dataset)}")

# Initialize FM
fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24, dropout=0.1)
optimizer = torch.optim.Adam(fm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"  Training for 50 epochs...")
train_losses = []

for epoch in range(50):
    fm.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = fm(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/50: Loss = {avg_loss:.6f}")

torch.save(fm.state_dict(), 'trained_fm_focused.pth')
print("  ✓ Saved trained_fm_focused.pth")

# ============================================================================
# STEP 4: HYPERPARAMETER SEARCH (Focused Grid)
# ============================================================================

print("\n[4] Focused hyperparameter search...")

param_grid = {
    'window_size': [5, 10, 20],
    'alpha_ridge': [0.1, 0.5, 1.0],
    'beta': [0.9, 0.95],
    'conf_alpha': [0.1]
}

results = []
best_mae = float('inf')
best_params = None

total_combinations = (len(param_grid['window_size']) * 
                     len(param_grid['alpha_ridge']) * 
                     len(param_grid['beta']))

print(f"  Testing {total_combinations} parameter combinations...")

# Use one validation dataset (SPY from training set)
val_data = train_datasets['SPY']['test']

count = 0
for ws in param_grid['window_size']:
    for alpha in param_grid['alpha_ridge']:
        for beta in param_grid['beta']:
            count += 1
            
            test_fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
            test_fm.load_state_dict(fm.state_dict())
            
            adapts = AdapTS(
                foundation_model=test_fm,
                seq_len=96,
                pred_len=24,
                alpha_ridge=alpha,
                window_size=ws,
                beta=beta,
                conf_alpha=0.1
            )
            
            res = adapts.evaluate_online(val_data[:400], verbose=False)
            
            results.append({
                'window_size': ws,
                'alpha_ridge': alpha,
                'beta': beta,
                'mae': res['mae'],
                'coverage': res['coverage']
            })
            
            if res['mae'] < best_mae:
                best_mae = res['mae']
                best_params = {'window_size': ws, 'alpha_ridge': alpha, 'beta': beta}
            
            print(f"    [{count}/{total_combinations}] ws={ws}, α={alpha}, β={beta} → MAE={res['mae']:.4f}")

print(f"\n  ✓ Best parameters: {best_params}")
print(f"    Validation MAE: {best_mae:.4f}")

# ============================================================================
# STEP 5: EVALUATE ON TRAINING SETS (Sanity Check)
# ============================================================================

print("\n[5] Evaluating on TRAINING datasets (sanity check)...")
train_results = {}

for ticker, data in train_datasets.items():
    test_fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
    test_fm.load_state_dict(fm.state_dict())
    
    adapts = AdapTS(
        foundation_model=test_fm,
        seq_len=96,
        pred_len=24,
        **best_params,
        conf_alpha=0.1
    )
    
    res = adapts.evaluate_online(data['test'], verbose=False)
    train_results[ticker] = res
    print(f"  {ticker}: MAE={res['mae']:.4f}, Coverage={res['coverage']*100:.1f}%")

# ============================================================================
# STEP 6: EVALUATE ON TESTING SETS (Generalization)
# ============================================================================

print("\n[6] Evaluating on TESTING datasets (unseen data)...")
test_results = {}

for ticker, data in test_datasets.items():
    test_fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
    test_fm.load_state_dict(fm.state_dict())
    
    adapts = AdapTS(
        foundation_model=test_fm,
        seq_len=96,
        pred_len=24,
        **best_params,
        conf_alpha=0.1
    )
    
    res = adapts.evaluate_online(data['test'], verbose=False)
    test_results[ticker] = res
    print(f"  {ticker}: MAE={res['mae']:.4f}, Coverage={res['coverage']*100:.1f}%")

# ============================================================================
# STEP 7: BASELINE COMPARISON
# ============================================================================

print("\n[7] Comparing with baseline (FM + Conformal only)...")
from adapts.uncertainty import ConformalPredictor

baseline_results = {}
for ticker, data in test_datasets.items():
    test_data = data['test']
    
    test_fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
    test_fm.load_state_dict(fm.state_dict())
    
    conf = ConformalPredictor(alpha=0.1, calibration_size=50)
    errors = []
    coverage_count = 0
    
    for i in range(min(100, len(test_data) - 120)):
        x = test_data[i:i+96]
        y_true = test_data[i+96:i+120]
        pred = test_fm.predict(x.reshape(1, -1)).flatten()
        result = conf.predict_with_interval(pred)
        conf.update_residuals(y_true, pred)
        errors.append(np.mean(np.abs(y_true - pred)))
        if np.all((y_true >= result['lower']) & (y_true <= result['upper'])):
            coverage_count += 1
    
    baseline_results[ticker] = {
        'mae': np.mean(errors),
        'coverage': coverage_count / len(errors)
    }
    print(f"  {ticker}: MAE={baseline_results[ticker]['mae']:.4f}")

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================

print("\n[8] Generating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Training loss curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_losses, linewidth=2, color='steelblue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Foundation Model Training Loss')
ax1.grid(True, alpha=0.3)

# Plot 2: Hyperparameter sensitivity
ax2 = fig.add_subplot(gs[0, 1])
for ws in param_grid['window_size']:
    ws_results = [r for r in results if r['window_size'] == ws]
    alphas = [r['alpha_ridge'] for r in ws_results]
    maes = [r['mae'] for r in ws_results]
    ax2.plot(alphas, maes, 'o-', label=f'ws={ws}', linewidth=2, markersize=8)
ax2.set_xlabel('Alpha (Ridge)')
ax2.set_ylabel('MAE')
ax2.set_title('Hyperparameter Sensitivity')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Training vs Testing performance
ax3 = fig.add_subplot(gs[1, :])
all_tickers = list(train_results.keys()) + list(test_results.keys())
train_maes = [train_results[t]['mae'] if t in train_results else 0 for t in all_tickers]
test_maes = [test_results[t]['mae'] if t in test_results else 0 for t in all_tickers]

x = np.arange(len(all_tickers))
width = 0.35
bars1 = ax3.bar(x[:len(train_results)] - width/2, train_maes[:len(train_results)], 
                width, label='Training Sets', color='lightblue')
bars2 = ax3.bar(x[len(train_results):] + width/2, test_maes[len(train_results):], 
                width, label='Testing Sets (Unseen)', color='coral')

ax3.set_xlabel('Dataset')
ax3.set_ylabel('MAE')
ax3.set_title('Performance: Training vs Testing Datasets')
ax3.set_xticks(x)
ax3.set_xticklabels(all_tickers, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: AdapTS vs Baseline comparison
ax4 = fig.add_subplot(gs[2, 0])
test_tickers = list(test_results.keys())
adapts_maes = [test_results[t]['mae'] for t in test_tickers]
baseline_maes = [baseline_results[t]['mae'] for t in test_tickers]

x = np.arange(len(test_tickers))
width = 0.35
ax4.bar(x - width/2, baseline_maes, width, label='Baseline (FM+Conf)', color='lightcoral')
ax4.bar(x + width/2, adapts_maes, width, label='AdapTS', color='lightgreen')

ax4.set_xlabel('Dataset')
ax4.set_ylabel('MAE')
ax4.set_title('AdapTS vs Baseline on Test Sets')
ax4.set_xticks(x)
ax4.set_xticklabels(test_tickers, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Summary statistics
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

train_avg_mae = np.mean([train_results[t]['mae'] for t in train_results])
train_avg_cov = np.mean([train_results[t]['coverage'] for t in train_results])
test_avg_mae = np.mean([test_results[t]['mae'] for t in test_results])
test_avg_cov = np.mean([test_results[t]['coverage'] for t in test_results])
baseline_avg_mae = np.mean([baseline_results[t]['mae'] for t in baseline_results])
improvement = ((baseline_avg_mae - test_avg_mae) / baseline_avg_mae) * 100

summary = f"""
FINAL RESULTS

Training Datasets: {len(train_datasets)}
  Avg MAE: {train_avg_mae:.4f}
  Avg Coverage: {train_avg_cov*100:.1f}%

Testing Datasets: {len(test_datasets)} (unseen)
  Avg MAE: {test_avg_mae:.4f}
  Avg Coverage: {test_avg_cov*100:.1f}%

Baseline (FM + Conformal):
  Avg MAE: {baseline_avg_mae:.4f}

Improvement: {improvement:.1f}% MAE reduction

Best Hyperparameters:
  Window Size: {best_params['window_size']}
  Alpha: {best_params['alpha_ridge']}
  Beta: {best_params['beta']}

Foundation Model:
  Epochs: 50
  Final Loss: {train_losses[-1]:.6f}
"""

ax5.text(0.1, 0.5, summary, fontsize=11, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.savefig('optimization_focused_results.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved optimization_focused_results.png")

# Save results
results_dict = {
    'best_params': best_params,
    'train_results': {k: {'mae': float(v['mae']), 'coverage': float(v['coverage'])} 
                     for k, v in train_results.items()},
    'test_results': {k: {'mae': float(v['mae']), 'coverage': float(v['coverage'])} 
                    for k, v in test_results.items()},
    'baseline_results': {k: {'mae': float(v['mae']), 'coverage': float(v['coverage'])} 
                        for k, v in baseline_results.items()},
    'improvement_pct': improvement
}

with open('optimization_focused_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("  ✓ Saved optimization_focused_results.json")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
print(f"Train on: {', '.join(TRAINING_TICKERS)}")
print(f"Test on: {', '.join(TESTING_TICKERS)}")
print(f"Improvement over baseline: {improvement:.1f}%")