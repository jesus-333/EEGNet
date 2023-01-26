# EEGNet

Implementation of EEGNet in PyTorch.

EEGNet is convolutional neural network proposed by Vernon J. Lawhern in the paper called **EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces**. As the name suggest is a convolutional network for EEG-based data.
The original paper can be read on [ArXiv](https://arxiv.org/abs/1611.08024) or [iopscience](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)

## Declare the network

### Use your parameters

To declare the network simply use:
```python
import EEGNet

model_config = dict(...)

model = EEGNet.EEGNet(model_config)
```

where ```model_config``` is a dictionary with all the parameters for the networks.

The name of the parameters follow the nomenclature of the original paper. So the list of possible parameter are:

|    Name    |                   Description                   |       Type       |
|:----------:|:-----------------------------------------------:|:----------------:|
|      C     |       Number of channels of the input EEG       |        int       |
|      T     |        Number of samples of the input EEG       |        int       |
|      D     |                Depth multipliers                |        int       |
|  n_classes |          Number of classes to classify          |        int       |
| c_kernel_1 | Dimension of the first kernel (temporal kernel) |       tuple      |
| c_kernel_2 | Dimension of the second kernel (spatial kernle) |       tuple      |
| c_kernel_3 |          Dimension of the third kernel          |       tuple      |
|  filter_1  |       Number of temporal filters (kernels)      |        int       |
|  filter_2  |       Number of spatial filters (kernels)       |        int       |
| p_kernel_1 |      Dimension of the first pooling kernel      |       tuple      |
| p_kernel_2 |     Dimensinon of the second pooling kernel     |       tuple      |
| activation |    Activation function (e.g. torch.nn.ELU())    | torch activation |
|  use_bias  |  Use or not the bias in the convolutional layer |       bool       |
|   dropout  |         Probability in the dropout layer        |       float      |

Here an example of a configuration dictionary with the same parameter of the original paper:

```python
model_config = dict(
    # EEG Parameters
    C = 22,
    T = 512,
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
```

### Use default parameters

If you not want to declare all the parameters I already created a function that automatically build the network with the defualt parameters of the paper. The only things you need to specify are the size of the EEG input.

```python
import EEGNet

model = EEGNet.get_model(C, T)
```

