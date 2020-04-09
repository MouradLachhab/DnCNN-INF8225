import torch.nn as nn

class DcNN(nn.Module):
    def __init__(self, nLayers=17):
        super().__init__()
        self.layers = []
        nb_features = 64

        # Input layer
        layers.append(nn.Cond2d()) 
        layers.append(nn.ReLU()) #nn.ReLU(inplace=True) ??

        # Hidden layers
        for i in range(0, nLayers):
            layers.append(nn.Cond2d())
            layers.append(nn.BatchNorm2d(nb_features))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Cond2d()) 

        self.model= nn.Sequential(*layers) #Runs all layer one by one

    def forward(self, image):
        return self.model(image)
