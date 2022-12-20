#%% Imports

import EEGNet
import moabb_dataset as md

import numpy as np
from torch.utils.data import DataLoader

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Main and train function

def main():
    # Get EEG data
    train_data, validation_data, test_data = get_data()
    
    # Get info about EEG data
    C = train_data[0][0].shape[1]
    T = train_data[0][0].shape[2]

    print("Number of channels         (C): ", C)
    print("Number of temporal samples (T): ", T)
    
    # Create untrained model
    model = get_model(C, T)

    return model


def train():
    pass


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Get function

def get_data():
    dataset_config = md.get_moabb_dataset_config()
    train_data, validation_data = md.get_train_data(dataset_config)
    test_data = md.get_test_data(dataset_config)
    
    return train_data, validation_data, test_data

def get_model(C, T):
    model_config = EEGNet.get_EEGNet_config(C, T)
    untrained_model = EEGNet.EEGNet(model_config)

    return untrained_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

if __name__ == "__main__":
    pass
