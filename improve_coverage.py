import numpy as np
import torch
from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts import AdapTS

print("="*70)
print("Improving Coverage to Target 90%")
print("="*70)

# Load test dataset
loader = TimeSeriesDataLoader(source='stock')
train_data, test_data = loader.load(ticker='GOOGL', start='2015-01-01', end='2024-01-01')

# Load trained FM
fm = BaselineFM(input_dim=1, hidden_dim=128, num_layers=2, pred_len=24)
fm.load_state_dict(torch.load('trained_fm_focused.pth'))

# Test different conformal alpha values
conf_alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
calibration_sizes = [50, 100, 200, 300]

print("\n[1] Testing conformal alpha values...")
for conf_alpha in conf_alphas:
    for calib_size in calibration_sizes:
        adapts = AdapTS(
            foundation_model=fm,
            seq_len=96,
            pred_len=24,
            window_size=5,
            alpha_ridge=0.1,
            beta=0.9,
            conf_alpha=conf_alpha,
            calibration_size=calib_size
        )
        
        res = adapts.evaluate_online(test_data[:300], verbose=False)
        print(f"  conf_α={conf_alpha:.2f}, calib={calib_size} → Coverage={res['coverage']*100:.1f}%, MAE={res['mae']:.4f}")

print("\n[2] Try conf_alpha=0.9 for 90% target coverage...")
adapts_high = AdapTS(
    foundation_model=fm,
    seq_len=96,
    pred_len=24,
    window_size=5,
    alpha_ridge=0.1,
    beta=0.9,
    conf_alpha=0.9,  # Flip it - want 90% coverage
    calibration_size=200
)

res = adapts_high.evaluate_online(test_data, verbose=True)
print(f"\nFinal: Coverage={res['coverage']*100:.1f}%, MAE={res['mae']:.4f}")