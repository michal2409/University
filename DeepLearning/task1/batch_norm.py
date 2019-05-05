import torch
from torch.nn.parameter import Parameter
from torch.nn import init

class Batch_norm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, training=True):
        super(Batch_norm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = training
        
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        
    def forward(self, X):
        assert len(X.shape) in (2, 4)
        
        # dense layer batch norm
        if len(X.shape) == 2:
            mean = torch.mean(X, 0)
            variance = torch.mean((X - mean) ** 2, 0)
            if self.training:
                X_norm = (X - mean) * 1.0 / torch.sqrt(variance + self.eps)
                self.running_mean = self.running_mean * self.momentum + mean * (1.0 - self.momentum)
                self.running_var = self.running_var * self.momentum + variance * (1.0 - self.momentum)
            else:
                X_norm = (X - self.running_mean) * 1.0 / torch.sqrt(self.running_var + self.eps)
            out = self.weight * X_norm + self.bias
        # conv layer batch norm
        else:
            B, C, H, W = X.shape

            mean = torch.mean(X, (0, 2, 3))
            variance = torch.mean((X - mean.reshape((1, C, 1, 1))) ** 2, (0, 2, 3))

            if self.training:
                X_norm = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.reshape((1, C, 1, 1)) + self.eps)
                self.running_mean = self.running_mean * self.momentum + mean * (1.0 - self.momentum)
                self.running_var = self.running_var * self.momentum + variance * (1.0 - self.momentum)
            else:
                X_norm=(X-self.running_mean.reshape((1,C,1,1)))*1.0/torch.sqrt(self.running_var.reshape((1,C,1,1))+self.eps)

            out = self.weight.reshape((1, C, 1, 1)) * X_norm + self.bias.reshape((1, C, 1, 1))

        return out