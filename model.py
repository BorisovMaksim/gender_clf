from torch.nn import init
import torch.nn as nn


class AudioClassifier(nn.Module):
    """ Audio Classification Model
            Baseline Architecture
            Input: (N, C, W, H) = (16, 1, 64, 282) - Spectrogram

            CONV2D -> MaxPOOL2D -> RELU ->  BatchNorm2D ->
            CONV2D -> MaxPOOL2D -> RELU ->  BatchNorm2D -> FLATTEN -> LINEAR -> LINEAR -> SIGMOID

    """

    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()

        conv_layers += [self.conv1, self.pool1, self.relu1, self.bn1, ]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()

        conv_layers += [self.conv2, self.pool2, self.relu2,  self.bn2]

        # Linear Layers
        self.lin1 = nn.Linear(in_features=816, out_features=300)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(0.75)
        self.lin2 = nn.Linear(in_features=300, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)


    def forward(self, x):
        """  Forward pass computations """
        x = self.conv(x)

        # Flatten
        x = x.view(x.shape[0], -1)

        x = self.lin1(x)
        x = self.relu2(x)
        x = self.drop1(x)

        x = self.lin2(x)
        x = self.sigmoid(x)

        return x
