import sys
sys.path.insert(0, '/Users/vidithiyer/adapts/ADAPTS')

from adapts.data import TimeSeriesDataLoader
from adapts.models import BaselineFM
from adapts import AdapTS

print("Testing AdapTS...")

loader = TimeSeriesDataLoader(source='stock')
train_data, test_data = loader.load(ticker='AAPL', start='2022-01-01', end='2024-01-01')

fm = BaselineFM(hidden_dim=64, pred_len=10)
adapts = AdapTS(foundation_model=fm, seq_len=50, pred_len=10)

results = adapts.evaluate_online(test_data[:200], verbose=True)

print(f"\nMAE: {results['mae']:.4f}")
print(f"Coverage: {results['coverage']*100:.1f}%")