from ddpm import NoiseScheduler, DDPM
import torch
from argparse import ArgumentParser
import numpy as np


# Define a function to remove noise from the samples
def removeNoise(xT : torch.Tensor, timeTensor : torch.Tensor, model : DDPM, noiseScheduler : NoiseScheduler):
    predNoise = model(xT, timeTensor)
    sqrtAlpha = torch.sqrt(noiseScheduler.alpha(timeTensor))[:, None]
    sqrtOneMinusAlpha = torch.sqrt(1 - noiseScheduler.alpha(timeTensor))[:, None]
    x0Pred = (xT - sqrtOneMinusAlpha * predNoise) / sqrtAlpha
    return x0Pred

    

# Parse the arguments from the command line
parser = ArgumentParser()
parser.add_argument('--model-path',
                    type = str,
                    default='models/trained_ddpm.pth',
                    help='Path to the trained model')
parser.add_argument('--prior-samples',
                    type = str,
                    default='data/albatross_prior_samples.npy',
                    help='Number of samples to generate from the prior')
parser.add_argument('--output-path',
                    type = str,
                    default='albatross_samples_reproduce.npy',
                    help='Path to save the generated samples')
arguments = parser.parse_args() 

# Load the trained model
model = DDPM(n_dim=64, n_steps=200)
model.load_state_dict(torch.load(arguments.model_path))
model.eval()

# Load the prior samples
priorSamples = torch.from_numpy(np.load(arguments.prior_samples)).to(torch.float32)
# Initialize the noise scheduler
noiseScheduler = NoiseScheduler(num_timesteps=200, beta_start=0.01, beta_end=1.0, type='linear')
# Generate samples using the deterministic sampling
numSamples = priorSamples.shape[0]
zT = priorSamples.clone()

# Reverse the diffusion process
xT = zT
for timestep in reversed(range(200)):
    timeTensor = torch.full((numSamples, 1), timestep, dtype=torch.long).to(device=zT.device)
    xT = removeNoise(xT, timeTensor, noiseScheduler, model)

samples = xT.detach().cpu().numpy()
np.save(arguments.output_path, samples)
print(f'Saved samples to {arguments.output_path}')