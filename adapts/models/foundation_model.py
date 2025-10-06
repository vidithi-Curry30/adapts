import torch
import torch.nn as nn
import numpy as np


class BaselineFM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2,
                 pred_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, pred_len)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self.forward(x).cpu().numpy()