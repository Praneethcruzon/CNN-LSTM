# Implementation is based on the architecture published in DeepVO Paper
# URL : https://arxiv.org/abs/1709.08429

import torch.nn as nn
from collections import OrderedDict

class CNN_LSTM(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels = 6):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initializing Convolutional Layers
        self.Conv1 = self._block(
            in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3, 
            stride = 2, name = "Conv 1", activation_needed = True
        )
        
        self.Conv2 = self._block(
            in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2,
            stride = 2, name = "Conv 2", activation_needed = True
        )

        self.Conv3 = self._block(
            in_channels = 128, out_channels = 256, kernel_size = 5, padding = 2,
            stride = 2, name = "Conv 3", activation_needed = True
        )

        self.Conv3_1 = self._block(
            in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1,
            stride = 1, name = "Conv 3_1", activation_needed = True
        )

        self.Conv4 = self._block(
            in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1,
            stride = 2, name = "Conv 4", activation_needed = True
        )

        self.Conv4_1 = self._block(
            in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1,
            stride = 1, name = "Conv 4_1", activation_needed = True
        )

        self.Conv5 = self._block(
            in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1,
            stride = 2, name = "Conv 5", activation_needed = True
        )

        self.Conv5_1 = self._block(
            in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1,
            stride = 1, name = "Conv 5_1", activation_needed = True
        )

        self.Conv6 = self._block(
            in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 1,
            stride = 2, name = "Conv 6", activation_needed = False
        )    
        # Pooling Layer
        self.max_pooling = nn.MaxPool2d(
            kernel_size = 2,
            stride = 1
        )

    def forward(self, x):
        print(x.shape)
        x = self.max_pooling(self.Conv1(x)); print(x.shape)
        x = self.max_pooling(self.Conv2(x)); print(x.shape)
        x = self.Conv3(x); print(x.shape)
        x = self.max_pooling(self.Conv3_1(x)); print(x.shape)
        x = self.Conv4(x); print(x.shape)
        x = self.max_pooling(self.Conv4_1(x)); print(x.shape)
        x = self.Conv5(x); print(x.shape)
        x = self.max_pooling(self.Conv5_1(x)); print(x.shape)
        x = self.Conv6(x); print(x.shape)

        # should add LSTM Layers

        return x

    def _block(self, in_channels, out_channels, kernel_size, padding, stride, name, activation_needed: bool):

        if activation_needed:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + f" - {kernel_size} x {kernel_size} Convolution",
                            nn.Conv2d(
                                in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                padding = padding,
                                stride = stride
                            )
                        ),
                        (
                            name + " - ReLU Activation",
                            nn.ReLU()
                        )
                    ]
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + f" {kernel_size} x {kernel_size} Convolution",
                            nn.Conv2d(
                                in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                padding = padding,
                                stride = stride
                            )
                        )
                    ]
                )
            )