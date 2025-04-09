import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #
import torch.nn as nn
from get_results import EnergyRegressor
import time


##########################################################

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    def __init__(self, model, stepSize = 0.1,
                numSamples = 1000, burnIn = 100, device='cpu'):
        '''
        model : PyTorch Model for computing the Energy E(X)
        stepSize : tau (Langevin Step Size)
        numSamples : total number of MCMC iterations
        burnIn : number of initial samples to discard
        '''
        self.model = model
        self.tau = stepSize
        self.N = numSamples
        self.burnIn = burnIn
        self.device = device

    def computeEnergy(self, input):
        '''
        Energy Function : E(X) = model(x)
        '''
        return self.model(input)


    def gradEnergy(self, input):
        '''
        Computes the gradient using the autograd function in torch
        '''
        input = input.clone().detach().requires_grad_(True)
        energy = self.computeEnergy(input).sum()
        gradient = torch.autograd.grad(energy, input)[0]
        return gradient

    def sample(self, x0):
        '''
        x0 : Initial sample with shape [batchSize, inputDim]
        Returns tensor of shape [numSamples - burnIn, inputDim]
        '''
        xT = x0.clone().detach().to(self.device)
        samples = []

        for t in range(self.N):
            gradientT = self.gradEnergy(xT)
            noise = torch.randn_like(xT)
            xProposed = xT - (self.tau ** 2 / 2) * gradientT + self.tau * noise

            # Compute the energy and the gradient at proposal
            Et = self.computeEnergy(xT).squeeze()
            Ep = self.computeEnergy(xProposed).squeeze()
            gradientP = self.gradEnergy(xProposed)

            # Compute the log proabilities of the proposal
            def logQ(xFrom, xTo, gradFrom):
                mean = xFrom - (self.tau ** 2 / 2) * gradFrom
                diff = xTo - mean
                return - (diff.pow(2).sum(dim=1)) / (2 * self.tau ** 2)

            logQxT_GivenXp = logQ(xProposed, xT, gradientP)
            logQxP_GivenXt = logQ(xT, xProposed, gradientT)

            # Applying the Metropolis Hastings Acceptance
            logAlpha = (Et - Ep + logQxT_GivenXp - logQxP_GivenXt).clamp(max = 80)
            alpha = torch.exp(logAlpha)
            u = torch.rand_like(alpha)
            
            accept = u < alpha
            xT = torch.where(accept.unsqueeze(1), xProposed, xT)

            if t >= self.burnIn:
                samples.append(xT.clone().detach())

        return torch.stack(samples) # Reshape into required dimensions

# class Algo2_Sampler:
    
# --- Main Execution ---
if __name__ == "__main__":
    modelPath = '/home/sedrica/CS726-Assignments/Assignment-4/DataFiles/trained_model_weights.pth'
    dataPath = '/home/sedrica/CS726-Assignments/Assignment-4/DataFiles/A4_test_data.pt'
    dataset = torch.load(dataPath)
    inputDim = dataset['x'].shape[1]
    model = EnergyRegressor(input_size = inputDim)
    print("[DEBUG] Loading Model Weights ...")
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    x0 = torch.randn(16, inputDim) # Taking the batch size to be 16

    sampler = Algo1_Sampler(model)
    print("[DEBUG] Sampler Sampling ...")
    startTime = time.time()
    samples = sampler.sample(x0) # Expected Shape : [800, 16, inputDim]
    print(f"[INFO] Time Taken : {time.time() - startTime}")
    print(f"[INFO] Datatype of Samples : {type(samples)}")
    print(f"[INFO] Size of Samples : {samples.shape}")
    outputPath = '/home/sedrica/CS726-Assignments/Assignment-4/DataFiles/OutputAlgo1.pt'
    torch.save(samples, outputPath)
    print("[DEBUG] Saving Samples to Output Path ...")
    

    