import torch.nn as nn

class DcNN(nn.Module):
    def __init__(self, nLayers=17):
        super().__init__()
        self.layers = []

        # Input layer
        layers.append(nn.Cond2d()) 

        # Hidden layers
        for i in range(0, nLayers):
            layers.append(nn.Cond2d())

        # Output layer
        layers.append(nn.Cond2d()) 

    def forward(self, image):
