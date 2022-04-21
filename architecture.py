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
        
        # self.flatten = nn.Flatten()

        # Output size of CNN layer [2, 1024, 5, 20]
        # Batch (stacked images), number of images, height, width of image.

        self.lstm = nn.LSTM(            
            input_size = 1024 * 5 * 20,
            hidden_size = 10, # As specified in paper. 
            batch_first = True,
            num_layers = 2
        )
        # self.lstm = nn.LSTM(1024 * 5 * 20, 10, 1, batch_first = True)
        self.fc1 = nn.Linear(10, 6)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128,64)
        # self.fc5 = nn.Linear(10,6)

    def CNN(self, x):

        x = self.max_pooling(self.Conv1(x)); print(x.shape)
        x = self.max_pooling(self.Conv2(x)); print(x.shape)
        x = self.Conv3(x); print(x.shape)
        x = self.max_pooling(self.Conv3_1(x)); print(x.shape)
        x = self.Conv4(x); print(x.shape)
        x = self.max_pooling(self.Conv4_1(x)); print(x.shape)
        x = self.Conv5(x); print(x.shape)
        x = self.max_pooling(self.Conv5_1(x)); print(x.shape)
        x = self.Conv6(x); print(x.shape)

        return x


    def forward(self, x):

        # Batch size default to 1
        num_frames, channels, height, width = x.shape
        print(num_frames, channels, height, width)

        
        # Reference : https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435" 
        # LSTM's input size should be -- input_size = 1024 * 5 * 20, batch_first = True

        cnn_input = x.view(num_frames, channels, height, width)
        print("CNN INPUT ", cnn_input.shape)
        cnn_out = self.CNN(cnn_input)
        print("CNN OUTPUT ", cnn_out.shape)

        lstm_in = cnn_out.view(1, num_frames, -1) # First Argument = 1 is the batch size
        print("LSTM INPUT ", lstm_in.shape)
        lstm_out, _ = self.lstm(lstm_in) # output will be in following format lstm_out, (hidden_state, cell_state)
        fc_1 = self.fc1(lstm_out[:,1])
        print("Forward - 1 \n",fc_1.shape)
        # fc_2 = self.fc2(fc_1)
        # print("Forward - 2 \n",fc_2.shape)
        # fc_3 = self.fc3(fc_2)
        # print("Forward - 3 \n",fc_3.shape)
        # fc_4 = self.fc4(fc_3)
        # print("Forward - 4 \n",fc_4.shape)
        # fc_5 = self.fc5(fc_4)
        # print("Forward - 5 \n",fc_5.shape)
        return fc_1     

        """ 
        # Reference : https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/Chapter10.ipynb
        # self.lstm = nn.LSTM(1024 * 5 * 20, 10, 1, batch_first = True)
        cnn_output = self.CNN(x[0])
        print(f"CNN output 1 : {cnn_output.view(1,1,1024*5*20).shape}")
        lstm_output,  (hn, cn) = self.lstm(cnn_output.view(1,1,1024*5*20))
        print(f"LSTM output 1 : {lstm_output.shape}")
        cnn_output = self.CNN(x[1])
        print(f"CNN output 2 : {cnn_output.view(1,1,1024*5*20).shape}")
        lstm_output, _ = self.lstm(cnn_output.view(1,1,1024*5*20))
        print(f"LSTM output 2 : {lstm_output.shape}")
        return lstm_output 
        """


        # return x

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