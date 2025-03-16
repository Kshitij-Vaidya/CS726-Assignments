import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from helperClasses import (TimeEmbedding, ResidualMLPModel, 
                           MLPModel, ConditionalMLPModel)

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.betaSchedule = torch.linspace(beta_start, beta_end, self.num_timesteps)
        self.alpha = 1 - self.betaSchedule
        self.alphaProd = torch.cumprod(self.alpha, 0)
        self.sqrtCumprodAlpha = np.sqrt(self.alphaProd)
        self.sqrtAlpha = np.sqrt(self.alpha)
        self.sqrtOneMinusAlpha = np.sqrt(1 - self.alpha)
        self.sqrtOneMinusAlphaProd = torch.sqrt(1 - self.alphaProd)

    def __len__(self):
        return self.num_timesteps
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.time_embed_dim = n_dim
        self.time_embed = TimeEmbedding(self.time_embed_dim)
        # self.model = MLPModel(n_dim, self.time_embed_dim)
        # self.model = AdvancedMLPModel(n_dim, self.time_embed_dim)
        self.model = ResidualMLPModel(n_dim, self.time_embed_dim)

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        # Get the time embeddings
        timeEmbeddings = self.time_embed(t)
        # Concatenate the input data with the time embeddings
        input = torch.cat([x, timeEmbeddings], dim=-1)
        # Get the predicted noise
        noise = self.model(input)
        return noise

class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200, n_classes=2):
        """
        Conditional Noise Prediction Network for DDPM with support for classifier-free guidance.
        
        Args:
            n_dim: int, dimensionality of the data
            n_steps: int, number of diffusion steps
            n_classes: int, number of classes (used for conditional generation)
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.time_embed_dim = n_dim
        # We use a class embedding of the same size as the data dimension
        self.class_embed_dim = n_dim
        self.time_embed = TimeEmbedding(self.time_embed_dim)
        self.class_embed = nn.Embedding(n_classes, n_dim)
        self.model = ConditionalMLPModel(n_dim, self.time_embed_dim, self.class_embed_dim)
    
    def forward(self, x, t, y=None):
        """
        Args:
            x: torch.Tensor, input data [batch_size, n_dim]
            t: torch.Tensor, timesteps [batch_size]
            y: torch.Tensor or None, class labels [batch_size] or None for unconditional
        Returns:
            torch.Tensor, predicted noise [batch_size, n_dim]
        """
        timeEmbeddings = self.time_embed(t)
        if y is None:
            # For unconditional prediction, use a zero vector as the class embedding.
            classEmbeddings = torch.zeros(x.size(0), self.n_dim, device=x.device)
        else:
            classEmbeddings = self.class_embed(y)
        # Concatenate data, class embedding, and time embedding.
        inp = torch.cat([x, classEmbeddings, timeEmbeddings], dim=-1)
        noise = self.model(inp)
        return noise
    pass
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler,
                 n_dim : int, n_classes : int, n_steps : int):
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.time_embed_dim = n_dim
        self.class_embed_dim = n_classes
        self.time_embed = TimeEmbedding(self.time_embed_dim)
        self.class_embed = nn.Embedding(n_classes, n_dim)
        self.model = ConditionalMLPModel(n_dim, self.time_embed_dim, self.class_embed_dim)

    def predict_proba(self, x):
        """
        Predict the class probabilities for input x by evaluating the reverse process
        conditioned on all possible classes.

        Args:
            x : (batch_size, inputDim) : Input tensor

        Returns:
            (batch_size, numClasses) : Probability distribution over classes
        """
        batch_size, inputDim = x.shape
        # Initialize a random noisy input (starting point for reverse diffusion)
        xT = torch.randn_like(x)
        logProbs = []

        # Loop over each possible class label
        for label in range(self.numClasses):
            # Initialize the conditioned input with the same noisy sample for this class
            inputs = xT.clone()

            # Reverse diffusion process over timesteps
            for timestep in reversed(range(self.noiseScheduler.num_timesteps)):
                # Create a tensor with the current timestep for all samples in the batch
                timesteps = torch.full((batch_size,), timestep, device=x.device)
                # Create a tensor with the current class label for all samples in the batch
                y = torch.full((batch_size,), label, device=x.device)
                # Predict noise conditioned on the current inputs, timestep, and class label
                noisePred = self.model(inputs, timesteps, y)
                # Update the inputs using the noise scheduler parameters
                inputs = ((inputs - ((1.0 - self.noiseScheduler.alpha[timestep]) / 
                                      self.noiseScheduler.sqrtOneMinusAlphaProd[timestep]) * noisePred)
                          / self.noiseScheduler.sqrtAlpha[timestep])
                
            # Compute a log-probability score using negative MSE between the denoised sample and the original input x
            logProb = -F.mse_loss(inputs, x, reduction='none').sum(dim=-1)
            logProbs.append(logProb)
        
        # Stack the scores and apply softmax to obtain probabilities
        logProbs = torch.stack(logProbs, dim=1)
        return F.softmax(logProbs, dim=1)
                
    def predict(self, x):
        """
        Predict the class for input x by evaluating the reverse process conditioned on all possible classes.

        Args:
            x : (batch_size, inputDim) : Input tensor

        Returns:
            (batch_size) : Predicted class indices
        """
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)
    
    

def train(model : DDPM, noise_scheduler : NoiseScheduler, 
          dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          epochs : int, run_name : str):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    model.train()
    lossFunction = nn.MSELoss()
    device = next(model.parameters()).device
    prevEpochLoss = -float('inf')
    bestModel = None
    for epoch in range(epochs):
        epochLoss = 0
        for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            # print(x)
            # Define the random time step
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, 
                                      (x.shape[0],), device=device)
            # Get the noise
            # print(timesteps)
            noise = torch.randn_like(x)
            noisyInput = (noise_scheduler.sqrtCumprodAlpha[timesteps, None] * x 
                          + noise_scheduler.sqrtOneMinusAlphaProd[timesteps, None] * noise)
            # print(noisyInput)
            optimizer.zero_grad()
            predictedNoise = model(noisyInput, timesteps)
            # print(predictedNoise)
            loss = lossFunction(predictedNoise, noise)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {epochLoss/len(dataloader)}")
        if epochLoss < prevEpochLoss:
            prevEpochLoss = epochLoss
            bestModel = model.state_dict()
    torch.save(bestModel, f'{run_name}/model.pth')

            

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   
    device = next(model.parameters()).device
    model.eval()
    samples = [] if return_intermediate else None
    numDim = model.model.inputDim
    inputs = torch.randn(n_samples, numDim, device=device)

    for timestep in reversed(range(0, noise_scheduler.num_timesteps)):
        timesteps = torch.full((n_samples,), timestep, device=device)
        noisePred = model(inputs, timesteps)
        inputs = ((inputs - ((1.0 - noise_scheduler.alpha[timestep]) / noise_scheduler.sqrtOneMinusAlphaProd[timestep]) * noisePred)
                  / noise_scheduler.sqrtAlpha[timestep])
        if return_intermediate:
            samples.append(inputs.clone().cpu().numpy())
    if return_intermediate:
        return samples
    return inputs

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, cond_dropout=0.2):
    """
    Train the conditional model with classifier-free guidance.
    
    Args:
        model: ConditionalDDPM, the model to train.
        noise_scheduler: NoiseScheduler, provides noise schedule parameters.
        dataloader: torch.utils.data.DataLoader, training data.
        optimizer: torch.optim.Optimizer, optimizer.
        epochs: int, number of epochs.
        run_name: str, name used for saving the trained model.
        cond_dropout: float, probability to drop the condition (simulate unconditional training).
    """
    # Move noise scheduler tensors to the correct device
    device = next(model.parameters()).device
    noise_scheduler.sqrtCumprodAlpha = noise_scheduler.sqrtCumprodAlpha.to(device)
    noise_scheduler.sqrtOneMinusAlphaProd = noise_scheduler.sqrtOneMinusAlphaProd.to(device)
    noise_scheduler.sqrtAlpha = noise_scheduler.sqrtAlpha.to(device)
    noise_scheduler.alpha = noise_scheduler.alpha.to(device)

    model.train()
    lossFunction = nn.MSELoss()
    for epoch in range(epochs):
        epochLoss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            # Random timesteps for each sample.
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            noisyInput = (noise_scheduler.sqrtCumprodAlpha[timesteps, None] * x +
                          noise_scheduler.sqrtOneMinusAlphaProd[timesteps, None] * noise)
            optimizer.zero_grad()
            # With probability cond_dropout, drop the condition.
            if torch.rand(1).item() < cond_dropout:
                y_input = None
            else:
                y_input = y
            predictedNoise = model(noisyInput, timesteps, y_input)
            loss = lossFunction(predictedNoise, noise)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {epochLoss/len(dataloader)}")
    torch.save(model.state_dict(), f'models/{run_name}.pth')

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label: int, guidance_scale=0.0):
    """
    Sample from the model using classifier-free guidance for a fixed class label.
    
    Args:
        model: ConditionalDDPM, the trained model.
        n_samples: int, number of samples to generate.
        noise_scheduler: NoiseScheduler, provides the noise schedule parameters.
        class_label: int, fixed class label for conditioning.
        guidance_scale: float, guidance strength (set >0 to steer sampling).
        return_intermediate: bool, if True returns all intermediate denoising steps.
    
    Returns:
        torch.Tensor of shape [n_samples, n_dim] with generated samples, or
        a list of intermediate steps if return_intermediate=True.
    """
    device = next(model.parameters()).device
    model.eval()
    numDim = model.n_dim
    inputs = torch.randn(n_samples, numDim, device=device)
    # Use a fixed class label for all samples
    y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)

    for timestep in reversed(range(noise_scheduler.num_timesteps)):
        timesteps = torch.full((n_samples,), timestep, device=device)
        if guidance_scale != 0.0:
            # Get unconditional prediction (simulate dropping the condition)
            unconditional_noise = model(inputs, timesteps, None)
            # Get conditional prediction
            conditional_noise = model(inputs, timesteps, y)
            # Combine the two predictions using the guidance formula
            noisePred = (1 + guidance_scale) * conditional_noise - guidance_scale * unconditional_noise
        else:
            noisePred = model(inputs, timesteps, y)
        
        # Update the denoising step using the noise scheduler parameters
        inputs = ((inputs - ((1.0 - noise_scheduler.alpha[timestep]) / 
                  noise_scheduler.sqrtOneMinusAlphaProd[timestep]) * noisePred)
                  / noise_scheduler.sqrtAlpha[timestep])
        
    return inputs


def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    device = next(model.parameters()).device
    model.eval()
    numDim = model.n_dim
    inputs = torch.randn(n_samples, numDim, device=device)
    # Use a fixed class label for all samples
    y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)

    for timestep in reversed(range(noise_scheduler.num_timesteps)):
        timesteps = torch.full((n_samples,), timestep, device=device)
        if guidance_scale != 0.0:
            # Get unconditional prediction (simulate dropping the condition)
            unconditional_noise = model(inputs, timesteps, None)
            # Get conditional prediction
            conditional_noise = model(inputs, timesteps, y)
            # Combine the two predictions using the guidance formula
            noisePred = (1 + guidance_scale) * conditional_noise - guidance_scale * unconditional_noise
        else:
            noisePred = model(inputs, timesteps, y)
        
        # Update the denoising step using the noise scheduler parameters
        inputs = ((inputs - ((1.0 - noise_scheduler.alpha[timestep]) / 
                  noise_scheduler.sqrtOneMinusAlphaProd[timestep]) * noisePred)
                  / noise_scheduler.sqrtAlpha[timestep])
        
    return inputs

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)
    # print(model)
    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        if args.dataset != 'albatross':
            data_y = data_y.to(device)
        else:
            data_y = torch.Tensor([0] * data_X.shape[0]).to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    else:
        raise ValueError(f"Invalid mode {args.mode}")