# AdapTS: Lightweight Adaptation for Time Series Foundation Models

Implementation of AdapTS - a system that enables time series foundation models to adapt in real-time without retraining.

## Installation
```bash
pip install -r requirements.txt
pip install -e .

from adapts import AdapTS
from adapts.data import TimeSeriesDataLoader

# Load data
loader = TimeSeriesDataLoader(source='stock')
train_data, test_data = loader.load(ticker='AAPL')

# Create AdapTS (you'll need a foundation model)
# adapts = AdapTS(foundation_model=your_model)



