import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self,channels, nLayers=17):
        super(DnCNN, self).__init__()
        layers = []
        kernel_size = 3
        padding = 1
        nb_features = 64

        layers.append(nn.Conv2d(in_channels=channels, out_channels=nb_features, kernel_size=kernel_size, padding=padding, bias=False)) 
        layers.append(nn.ReLU(inplace=True)) 

        for _ in range(0, nLayers - 2):
            layers.append(nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(nb_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=nb_features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)) 

        self.model= nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)