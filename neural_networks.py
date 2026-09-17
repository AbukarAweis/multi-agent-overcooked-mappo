import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=96, output_dim=6, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.norm1(self.in_layer(x)))
        x = F.relu(self.norm2(self.hidden_layer(x)))
        x = self.out_layer(x)
        return F.log_softmax(x, dim=-1)

class ValueFunctionNetwork(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128):
        super(ValueFunctionNetwork, self).__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.norm1(self.in_layer(x)))
        x = F.relu(self.norm2(self.hidden_layer(x)))
        return self.out_layer(x)

