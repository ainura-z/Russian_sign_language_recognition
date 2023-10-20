import torch
import torch.nn as nn


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers, bias=True, batch_first=True)  
        self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size*2, num_layers=num_layers, bias=True, batch_first=True)
        self.lstm_3 = nn.LSTM(self.hidden_size*2, self.hidden_size, num_layers=num_layers, bias=True, batch_first=True)
        self.dense_1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.dropout = nn.Dropout(p=0.2)
        self.dense_2 = nn.Linear(int(self.hidden_size/2), num_classes)
        
    def forward(self, x): 
        out_1, (hidden_1, cell_1) = self.lstm_1(x.float())
        out_2, (hidden_2, cell_2) = self.lstm_2(out_1[:,-1,:].unsqueeze(1)) #(torch.squeeze(hidden_1).unsqueeze(1))
        out_3, (hidden_3, cell_3) = self.lstm_3(out_2[:,-1,:].unsqueeze(1)) #(torch.squeeze(hidden_2).unsqueeze(1))

        
        # Classification
        output_dense_1 = nn.functional.relu(self.dense_1(out_3[:,-1,:]))
        output_dense_1 = self.dropout(output_dense_1)
        pred = self.dense_2(output_dense_1)

        
        return nn.functional.log_softmax(pred)