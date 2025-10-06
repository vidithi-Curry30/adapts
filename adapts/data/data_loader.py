import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataLoader:
    def __init__(self, source: str = 'stock'):
        self.source = source
        self.scaler = StandardScaler()
        
    def load_stock_data(self, ticker: str = 'AAPL', start: str = '2020-01-01',
                       end: str = '2024-01-01', column: str = 'Close') -> np.ndarray:
        print(f"Downloading {ticker} data from {start} to {end}...")
        data = yf.download(ticker, start=start, end=end, progress=False)
        values = data[column].values.reshape(-1, 1)
        print(f"Loaded {len(values)} data points")
        return values
    
    def load_csv(self, filepath: str, column: str) -> np.ndarray:
        df = pd.read_csv(filepath)
        values = df[column].values.reshape(-1, 1)
        return values
    
    def load(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self.source == 'stock':
            data = self.load_stock_data(**kwargs)
        elif self.source == 'csv':
            data = self.load_csv(**kwargs)
        elif self.source == 'array':
            data = kwargs.get('data')
        else:
            raise ValueError(f"Unknown source: {self.source}")
        
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        self.scaler.fit(train_data)
        train_normalized = self.scaler.transform(train_data)
        test_normalized = self.scaler.transform(test_data)
        
        return train_normalized.flatten(), test_normalized.flatten()
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()