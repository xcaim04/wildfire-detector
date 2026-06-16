import torch
import torch.nn as nn

class WildfireNet(nn.Module):
    
    def __init__(self, input_dim=12):
        super(WildfireNet, self).__init__()

        self.network = nn.Sequential(
            # Capa 1: Entran 12 variables climáticas y salen 8
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            # Capa 2: Entran 8 (del paso anterior) y salen 4
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            # Capa 3: Entran 4 (del paso anterior) y sale 1 probabilidad
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)