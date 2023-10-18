import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, hidden_layer_sizes):
        super().__init__()
        input_dim = 108
        layers = []
        for i in range(len(hidden_layer_sizes)):
            layers.append(nn.Linear(input_dim, hidden_layer_sizes[i]))
            layers.append(nn.GELU())
            input_dim = hidden_layer_sizes[i]
        layers.append(nn.Linear(input_dim, 2))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)