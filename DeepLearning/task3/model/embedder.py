import torch
from torch import nn

class embedder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embd_dim):
        super(embedder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, embd_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        out, _ = self.lstm(x, (h0, c0))
        out = torch.mean(out, 1).unsqueeze(0)
        return self.fc(out)