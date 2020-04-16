import torch.nn as nn

# TODO: Remove useless comments once done with the project

class DnCNN(nn.Module):
    def __init__(self,channels, nLayers=17):
        super(DnCNN, self).__init__()
        layers = []
        kernel_size = 3
        padding = 1
        nb_features = 64

        # Input layer
        # Bias False is used as they say it has very little effect on bigger networks for convolutions
        layers.append(nn.Cond2d(in_channels=channels, out_channels=nb_features, kernel_size=kernel_size, padding=padding, bias=False)) 
        layers.append(nn.ReLU(inplace=True)) # I put it in cause its in the original code, but some say online that it is not always good to use. Not entirely sure why

        # Hidden layers
        for i in range(0, nLayers):
            layers.append(nn.Cond2d(in_channels=nb_features, out_channels=nb_features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(nb_features))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Cond2d(in_channels=nb_features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)) 

        self.model= nn.Sequential(*layers) #Runs all layer one by one

    def forward(self, image):
        return self.model(image)
