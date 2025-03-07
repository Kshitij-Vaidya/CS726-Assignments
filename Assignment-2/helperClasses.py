import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    def __init__(self, numChannels: int):
        super().__init__()
        # numChannels is the number of channels in the input
        self.numChannels = numChannels
        # Define the layers of the Multi Layer Perceptron (MLP) Model
        self.linear1 = nn.Linear(self.numChannels // 4, 
                                 self.numChannels)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(self.numChannels, 
                                 self.numChannels)
    
    def forward(self, timeInput: torch.Tensor):
        halfDim = self.numChannels // 8
        embeddings = math.log(10000) / (halfDim - 1)
        embeddings = torch.exp(torch.arange(halfDim) * -embeddings).to(timeInput.device)
        embeddings = timeInput[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # Pass the embeddings through the linear and activation layers
        embeddings = self.linear1(embeddings)
        embeddings = self.activation(embeddings)
        embeddings = self.linear2(embeddings)
        # Return the final embeddings of the time input
        return embeddings


# Define the Residual Block class that has convolution layers with group normalisation
class ResidualBlock(nn.Module):
    def __init__(self, inputChannels: int, outputChannels: int,
                 timeChannels: int, numGroups: int = 32, dropout: float = 0.1):
        super().__init__()
        # Define group normalisation and the first layer
        self.norm1 = nn.GroupNorm(numGroups, inputChannels)
        self.activation1 = nn.SiLU()
        self.convLayer1 = nn.Conv2d(inputChannels, outputChannels,
                                    kernel_size=3, padding=1)
        # Define the second layer with group normalisation
        self.norm2 = nn.GroupNorm(numGroups, outputChannels)
        self.activation2 = nn.SiLU()
        self.convLayer2 = nn.Conv2d(outputChannels, outputChannels,
                                    kernel_size=3, padding=1)
        
        # If the input and output layers are not the same, then use a shortcut connection
        if inputChannels != outputChannels:
            self.shortcut = nn.Conv2d(inputChannels, outputChannels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Add the linear layer for the time embeddings
        self.timeEmbedding = nn.Linear(timeChannels, outputChannels)
        self.timeActivation = nn.SiLU()
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, time: torch.Tensor):
        # Passing the input through the first layer
        outLayer1 = self.norm1(input)
        outLayer1 = self.activation1(outLayer1)
        outLayer1 = self.convLayer1(outLayer1)
        # Add the time embeddings to the output of the first layer
        outLayer1 = outLayer1 + self.timeActivation(self.timeEmbedding(time))[:, :, None, None]
        # Pass the output through the second layer
        outLayer2 = self.norm2(outLayer1)
        outLayer2 = self.activation2(outLayer2)
        outLayer2 = self.dropout(outLayer2)
        outLayer2 = self.convLayer2(outLayer2)
        # Add the shortcut connection
        output = outLayer2 + self.shortcut(input)
        return output
    

# Define the Attention Block 
class AttentionBlock(nn.Module):
    def __init__(self, numChannels: int, numHeads: int = 1, headDim: int = None, numGroups = 32):
        super().__init__()
        if headDim is None:
            headDim = numChannels
        # Define the normalisation layer
        self.norm = nn.GroupNorm(numGroups, numChannels)

        self.projection = nn.Linear(numChannels, numHeads * headDim * 3)

        self.output = nn.Linear(numHeads * headDim, numChannels)
        # Scale for dot product attention
        self.scale = headDim ** -0.5
        self.numHeads = numHeads
        self.headDim = headDim

    def forward(self, input: torch.Tensor, time: Optional[torch.Tensor] = None):
        batchSize, numChannels, height, width = input.shape 
        input = input.view(batchSize, numChannels, -1).permute(0, 2, 1)
        input = self.projection(input).view(batchSize, -1, self.numHeads, self.headDim * 3)

        query, key, value = input.chunk(3, dim=-1)
        attention = torch.einsum('b i h d, b j h d -> b i j h', query, key) * self.scale
        attention = F.softmax(attention, dim=2)

        result = torch.einsum('b i j h, b j h d -> b i h d', attention, value)
        result = result.reshape(batchSize, -1, self.numHeads * self.headDim)
        result = self.output(result)
        # Adding the skip connection
        output = result + input
        output = output.permute(0, 2, 1).view(batchSize, numChannels, height, width)
        return output
    


if __name__ == '__main__':
    # Instantiate and test the TimeEmbedding module
    timeEmbed = TimeEmbedding(numChannels=128)
    test_time_input = torch.randn(4)
    time_output = timeEmbed(test_time_input)
    print("Time embedding output shape:", time_output.shape)

    # Instantiate and test the ResidualBlock
    resBlock = ResidualBlock(32, 64, 128)
    res_input = torch.randn(1, 32, 16, 16)
    res_time = torch.randn(1, 128)
    res_output = resBlock(res_input, res_time)
    print("Residual block output shape:", res_output.shape)

    # Instantiate and test the AttentionBlock
    attBlock = AttentionBlock(numChannels=64, numHeads=2, headDim=32)
    att_output = attBlock(res_output)
    print("Attention block output shape:", att_output.shape)