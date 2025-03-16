import torch.utils
import torch.utils.data
from ddpm import (DDPM, NoiseScheduler, ClassifierDDPM,
                  train, sample)
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_emd, get_likelihood, split_data
from dataset import load_dataset
import os
import utils


def modelRun(model : DDPM, noiseScheduler : NoiseScheduler, 
             optimizer : torch.optim.Optimizer, 
             trainDataloader : torch.utils.data.DataLoader,
             testDataloader : torch.utils.data.DataLoader, 
             numEpochs : int, run_name : str):
    '''
    Function accepts an untrained model, trains it on the given dataset
    Make an equal number of samples as the dataset
    Return the EMD, Log Likelihood of the dataset and the samples
    '''
    # Train the model using the given dataloader
    numSamples = len(trainDataloader.dataset)
    train(model, noiseScheduler, trainDataloader, optimizer, numEpochs, run_name)
    # Sample from the model
    trainedModel = DDPM(n_dim=model.n_dim, n_steps=model.n_steps)
    # print(os.path.exists(f'{run_name}/model.pth'))
    # print('[TEST RUN] Loading ...')
    # trainedModel.load_state_dict(torch.load(f'{run_name}/model.pth'))
    model_path = f'{run_name}/model.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")

    if state_dict is None:
        raise ValueError(f"Loaded state_dict is None! Possible corrupted file: {model_path}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    trainedModel = DDPM(n_dim=model.n_dim, n_steps=model.n_steps)
    trainedModel.load_state_dict(state_dict)

    print("Model loaded successfully!")
    samples = sample(trainedModel, numSamples, noiseScheduler)
    # Calculate the EMD and Log Likelihood of the dataset and the samples
    datasetX, datasetY = trainDataloader.dataset.tensors
    datasetX = datasetX.detach().cpu()
    datasetY = datasetY.detach().cpu()
    samples = samples.detach().cpu()
        # Calculate the EMD and Log Likelihood of the test set and the samples
    testX, testY = testDataloader.dataset.tensors
    testX = testX.detach().cpu()
    testY = testY.detach().cpu()
    # trainEMD = get_emd(datasetX, samples)
    subsample_size = 600
    train_emd_list = []
    test_emd_list = []
    for i in range(10):
        subsample_test_X = utils.sample(testX, size = subsample_size)
        subsample_train_X = utils.sample(datasetX, size = subsample_size)
        subsample_samples = utils.sample(samples, size = subsample_size)
        test_emd =  utils.get_emd(subsample_test_X.numpy(), subsample_samples.numpy())
        train_emd = utils.get_emd(subsample_train_X.numpy(), subsample_samples.numpy())
        print(f'{i} EMD w.r.t test split : {test_emd: .3f}')
        print(f'{i} EMD w.r.t train split: {train_emd: .3f}')
        train_emd_list.append(train_emd)
        test_emd_list.append(test_emd)
    trainLL = get_likelihood(datasetX, samples, temperature=0.1)
    trainEMD = np.mean(train_emd_list)
    testEMD = np.mean(test_emd_list)
    # testEMD = get_emd(testX, samples)
    testLL = get_likelihood(testX, samples, temperature=0.1)
    return trainEMD, trainLL, testEMD, testLL


def timestepDependence(dataset : str):
    '''
    Function to study the dependence of the EMD and Log Likelihood on the number of timesteps
    '''
    # Define the parameters
    numEpochs = 50
    numTimesteps = [10, 50, 100, 150, 200]
    # Load the dataset
    dataX, dataY = load_dataset(dataset)
    trainX, trainY, testX, testY = split_data(dataX, dataY)
    trainDataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trainX, trainY), batch_size=64, shuffle=True)
    testDataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(testX, testY), batch_size=64, shuffle=True)
    trainEMDList, testEMDList, trainLLList, testLLList = [], [], [], []
    # Define the model
    for numTimestep in numTimesteps:
        print(len(trainX[0]))
        model = DDPM(n_dim=len(trainX[0]), n_steps=numTimestep)
        noiseScheduler = NoiseScheduler(num_timesteps=numTimestep, beta_start=0.01, beta_end=0.99, type='linear')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        run_name = f'exps/ddpm_{numTimestep}'
        os.makedirs(run_name, exist_ok=True)
        trainEMD, trainLL, testEMD, testLL = modelRun(model, noiseScheduler, optimizer, trainDataloader, testDataloader, numEpochs, run_name)
        trainEMDList.append(trainEMD)
        trainLLList.append(trainLL)
        testEMDList.append(testEMD)
        testLLList.append(testLL)
        print(f'Number of Timesteps: {numTimestep}')
        print(f'Training EMD: {trainEMD}, Training Log Likelihood: {trainLL}')
        print(f'Test EMD: {testEMD}, Test Log Likelihood: {testLL}')
        print('--------------------------------------')
    # Plot the EMD and LL values and save the figures
    plt.figure()
    plt.plot(numTimesteps, trainEMDList, label='Training EMD')
    plt.plot(numTimesteps, testEMDList, label='Test EMD')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('EMD')
    plt.legend()
    plt.savefig(f'{dataset}_timestep_dependence_emd.png')
    plt.figure()
    plt.plot(numTimesteps, trainLLList, label='Training Log Likelihood')
    plt.plot(numTimesteps, testLLList, label='Test Log Likelihood')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.savefig(f'{dataset}_timestep_dependence_ll.png')


if __name__ == '__main__':
    timestepDependence('moons')