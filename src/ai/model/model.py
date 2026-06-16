import torch
import torch.nn as nn

class WildfireNet(nn.Module):
    
    def __init__(self, input_dim=12):
        super(WildfireNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)