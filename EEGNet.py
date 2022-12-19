import torch
from torch import nn

class EEGNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        use_bias = config['use_bias']
        C = config['C']
        T = config['T']
        D = config['D']

        self.temporal_filter = nn.Sequential(
            nn.Conv2d(1, config['filter_1'], kernel_size = config['c_kernel_1'], padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_1']),
        )

        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], group = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_1']),
            nn.Dropout(config['dropout'])
        )

        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = config['c_kernel_3'], group = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_2'], config['filter_2'], kernel_size = (1, 1), group = 1, padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_2']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        x = self.separable_convolution(x)

        return x


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def config_EEGNet(C = 22, T = 512):
    config = dict(
        # EEG Parameters
        C = C,
        T = T,
        D = 8,
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
        use_biase = False,
        dropout = 0.5
    )

    return config
