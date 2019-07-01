from torch import nn
from .embedder import embedder
from .brnn import brnn

class net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embed_dim=100):
        super(net, self).__init__()
        self.embedder = embedder(input_size, hidden_size, num_layers, embed_dim)
        self.brnn = brnn(embed_dim, hidden_size, num_layers, num_classes, embed_dim)
    
    def forward(self, x, y):
        return self.brnn(self.embedder(x), self.embedder(y))
    
    def get_embed(self):
        return self.embedder