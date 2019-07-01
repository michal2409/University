import torch
import torch.nn.functional as F
from torch import nn

class brnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embed_dim):
        super(brnn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_size+embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, y):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        out, _ = self.lstm(x, (h0, c0))
        
        out = torch.mean(out, 1)
        out = torch.cat([out, y[:, -1, :]], dim=1)
        out = self.fc2(self.fc1(out))
        return F.log_softmax(out, dim=1)