"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Training script for EEGNet
"""

"""
%load_ext autoreload
%autoreload 2

import moabb_dataset as md
import train
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import EEGNet

import numpy as np
import matplotlib.pyplot as plt
import moabb_dataset as md

import torch
from torch import optim, nn
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

    # Get train config
    train_config = get_train_config()
    
    # Create DataLoader
    train_loader = DataLoader(train_data, train_config['batch_size'])
    validation_loader = DataLoader(validation_data, train_config['batch_size'])
    test_loader = DataLoader(test_data, train_config['batch_size'])
    loader_list = [train_loader, validation_loader]

    # Train model
    model = train(model, loader_list, train_config)

    return model


def train(model, loader_list, config):
    # Get train/validaton dataloader
    train_loader = loader_list[0]
    validation_loader = loader_list[1]

    # Move model to training device (cpu/gpu)
    model.to(config['device'])

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr = config['lr'], 
                                  weight_decay = config['optimizer_weight_decay'])

    # lr scheduler
    if config['use_lr_scheduler'] == True:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['lr_decay_rate'])
    else:
        lr_scheduler = None

    # Loss function
    loss_function = nn.NLLLoss()

    # Tracking variables
    train_loss_list = []
    validation_loss_list = []
    fig, ax = plt.subplots(figsize = config['loss_figsize'])
    plt.show()

    for epoch in range(config['epochs']):
        # Train the model
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, config)
        
        # Validation
        validation_loss = validation_epoch(model, validation_loader, loss_function, config)
        
        # Lr scheduler step
        if lr_scheduler is not None: lr_scheduler.step()

        # Print loss
        print_loss(epoch, train_loss, validation_loss)

        # Save and visualize loss
        train_loss_list.append(train_loss)
        validation_loss_list.append(validation_loss)
        visualize_loss(fig, ax, train_loss_list, validation_loss_list)

        
def train_epoch(model, loader, loss_function, optimizer, config):
    model.train()
    
    total_loss = 0
    for batch in loader:
        x = batch[0].to(config['device'])
        y_true = batch[1].to(config['device'])
        
        # Forward step
        y_predict = model(x)
        
        # Loss computation
        train_loss = loss_function(y_predict, y_true)
        
        # Zero past gradients
        optimizer.zero_grad()

        # Backward and optimization step
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item() * x.shape[0]

    total_loss /= len(loader.sampler)
    
    return total_loss

def validation_epoch(model, loader, loss_function, config):
    model.eval()
    
    with torch.no_grad():
        total_loss = 0
        for batch in loader:
            x = batch[0].to(config['device'])
            y_true = batch[1].to(config['device'])
            
            # Forward step
            y_predict = model(x)
            
            # Loss computation
            validation_loss = loss_function(y_predict, y_true)
            
            total_loss += validation_loss.item() * x.shape[0]

        total_loss /= len(loader.sampler)

    return total_loss

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


def get_train_config():
    train_config = dict(
        batch_size = 32,
        epochs = 100,
        lr = 1e-3,
        use_lr_scheduler = True,
        lr_decay_rate = 0.995,
        optimizer_weight_decay = 1e-3,
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        loss_figsize = (15, 10)
    )

    return train_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Visualization function

def print_loss(epoch, train_loss, validation_loss):
    print("Epoch:{}".format(epoch))
    print("\tTrain Loss     : ", train_loss)
    print("\tValidation Loss: ", validation_loss, "\n")

def visualize_loss(fig, ax, train_loss, validation_loss):
    x = np.arange(len(train_loss))

    ax.plot(x, train_loss)
    ax.plot(x, validation_loss)

    ax.legend(["Train", "Validation"])

    # fig.show()
    # plt.draw()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

if __name__ == "__main__":
    main()
