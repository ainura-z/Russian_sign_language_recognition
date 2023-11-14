import torch
import torch.nn as nn


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(input_size, self.hidden_size,
                              bias=True, batch_first=True)  
        self.dense_1 = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, x):     
        outputs1, (hidden_1, cell_1) = self.lstm_1(x)
        hidden = hidden_1.view(-1, self.hidden_size)
        
        # Classification
        pred = self.dense_1(hidden)
    
        return pred