"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Definition of EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Network declaration

class EEGNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        use_bias = config['use_bias']
        D = config['D']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section

        self.temporal_filter = nn.Sequential(
            nn.Conv2d(1, config['filter_1'], kernel_size = config['c_kernel_1'], padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_1']),
        )
        
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_1'] * D),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_1']),
            nn.Dropout(config['dropout'])
        )

        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_2']),
            nn.Dropout(config['dropout'])
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feedforward
        
        # Compute the number of input neurons of the feedforward layer
        x = torch.rand(1,1,config['C'], config['T'])
        x = self.separable_convolution(self.spatial_filter(self.temporal_filter(x)))
        input_neurons = len(x.flatten())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_neurons, config['n_classes']),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        x = self.separable_convolution(x)
        x = self.classifier(x)

        return x


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# EEGNet config

def get_EEGNet_config(C = 22, T = 512):
    config = dict(
        # EEG Parameters
        C = C,
        T = T,
        D = 2,
        n_classes = 4,
        # Convolution: kernel size
        c_kernel_1 = (1, 64),
        c_kernel_2 = (C, 1),
        c_kernel_3 = (1, 16),
        # Convolution: number of filter
        filter_1 = 8,
        filter_2 = 16,
        #Pooling kernel
        p_kernel_1 = (1, 4),
        p_kernel_2 = (1, 8),
        # Other parameters
        activation = nn.ELU(),
        use_bias = False,
        dropout = 0.5
    )

    return config
