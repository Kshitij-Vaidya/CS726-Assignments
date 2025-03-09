import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, List


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
                 timeChannels: int, numGroups: int = 4, dropout: float = 0.1):
        super().__init__()
        # Define group normalisation and the first layer
        print(f'Input Channels : {inputChannels}')
        print(f'Output Channels : {outputChannels}')
        print(f'Number of Groups : {numGroups}')
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
        outLayer1 = outLayer1 + self.timeEmbedding(self.timeActivation(time))[:, :, None, None]
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
        print(f'Input Shape : {input.shape}')
        batchSize, numChannels, height, width = input.shape 
        input = input.view(batchSize, numChannels, -1).permute(0, 2, 1)
        temp = self.projection(input).view(batchSize, -1, self.numHeads, self.headDim * 3)

        query, key, value = torch.chunk(temp, 3, dim=-1)
        attention = torch.einsum('b i h d, b j h d -> b i j h', query, key) * self.scale
        attention = F.softmax(attention, dim=2)

        result = torch.einsum('b i j h, b j h d -> b i h d', attention, value)
        print(f'Result Shape : {result.shape}')
        result = result.reshape(batchSize, -1, self.numHeads * self.headDim)
        print(f'Reshaped Result Shape : {result.shape}')
        result = self.output(result)
        print(f'Result after Output : {result.shape}')
        # Adding the skip connection
        output = result + input
        output = output.permute(0, 2, 1).view(batchSize, numChannels, height, width)
        return output


class DownSampleBlock(nn.Module):
    def __init__(self, inputChannels: int, outputChannels, timeChannels: int, hasAttention: bool):
        super().__init__()
        self.residualBlock = ResidualBlock(inputChannels, outputChannels, timeChannels)

        if hasAttention:
            self.attentionBlock = AttentionBlock(outputChannels)
        else:
            self.attentionBlock = nn.Identity()
        
    def forward(self, input: torch.Tensor, time: torch.Tensor):
        input = self.residualBlock(input, time)
        output = self.attentionBlock(input)
        return output


class UpSampleBlock(nn.Module):
    def   __init__(self, inputChannels: int, 
                   outputChannels: int, 
                   timeChannels: int, hasAttention: bool):
        super().__init__()
        self.residualBlock = ResidualBlock(inputChannels, outputChannels, timeChannels)
        if hasAttention:
            self.attentionBlock = AttentionBlock(outputChannels)
        else:
            self.attentionBlock = nn.Identity()
    
    def forward(self, input: torch.Tensor, time: torch.Tensor):
        input = self.residualBlock(input, time)
        output = self.attentionBlock(input)
        return output


class MiddleBlock(nn.Module):
    def __init__(self, numChannels: int, timeChannels: int):
        super().__init__()
        self.residualBlock1 = ResidualBlock(numChannels, numChannels, timeChannels)
        self.attentionBlock = AttentionBlock(numChannels)
        self.residualBlock2 = ResidualBlock(numChannels, numChannels, timeChannels)
    
    def forward(self, input: torch.Tensor, time: torch.Tensor):
        input = self.residualBlock1(input, time)
        input = self.attentionBlock(input)
        output = self.residualBlock2(input, time)
        return output


class DownSample(nn.Module):
    def __init__(self, numChannels):
        super().__init__()
        self.conv = nn.Conv2d(numChannels, numChannels * 2, 
                              kernel_size=3, stride=2, padding=1)
    
    def forward(self, input: torch.Tensor):
        return self.conv(input)


class UpSample(nn.Module):
    def __init__(self, numChannels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(numChannels, numChannels // 2, 
                                       kernel_size=4, stride=2, padding=1)
    
    def forward(self, input: torch.Tensor):
        return self.conv(input)
    

class UNetModel(nn.Module):
    def __init__(self, imageChannels: int = 3, numChannels: int = 8,
                 channelMultiplier: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 isAttention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 numBlocks: int = 2):
        super().__init__()
        numResolutions = len(channelMultiplier)
        self.imageProjection = nn.Conv2d(imageChannels, numChannels, kernel_size=3, padding=1)
        self.timeEmbedding = TimeEmbedding(numChannels * 4)
        downSampleBlocks = []

        outputChannels = inputChannels = numChannels
        for i in range(numResolutions):
            outputChannels = inputChannels * channelMultiplier[i]
            for _ in range(numBlocks):
                downSampleBlocks.append(DownSampleBlock(inputChannels, outputChannels, numChannels * 4, isAttention[i]))
                inputChannels = outputChannels
            if i < numResolutions - 1:
                downSampleBlocks.append(DownSample(inputChannels))
        
        self.downSampleBlocks = nn.ModuleList(downSampleBlocks)
        self.middleBlock = MiddleBlock(outputChannels, numChannels * 4, )

        upSampleBlocks = []
        inputChannels = outputChannels
        for i in reversed(range(numResolutions)):
            outputChannels = inputChannels // channelMultiplier[i]
            for _ in range(numBlocks):
                upSampleBlocks.append(UpSampleBlock(inputChannels, outputChannels, numChannels * 4, isAttention[i]))
                inputChannels = outputChannels
            if i > 0:
                upSampleBlocks.append(UpSample(inputChannels))
        
        self.upSampleBlocks = nn.ModuleList(upSampleBlocks)

        self.groupNorm = nn.GroupNorm(8, numChannels)
        self.activation = nn.SiLU()
        self.outputLayer = nn.Conv2d(numChannels, imageChannels, kernel_size=3, padding=1)
    
    def forward(self, image: torch.Tensor, time: torch.Tensor):
        image = self.imageProjection(image)
        time = self.timeEmbedding(time)
        skipConnectionOutputs = [image]

        for downSampleBlock in self.downSampleBlocks:
            image = downSampleBlock(image, time)
            skipConnectionOutputs.append(image)
        
        image = self.middleBlock(image, time)

        for upSampleBlock in self.upSampleBlocks:
            if isinstance(upSampleBlock, UpSample):
                image = upSampleBlock(image, time)
            else:
                skipConnection = skipConnectionOutputs.pop()
                image = torch.cat([image, skipConnection], dim=1)
                image = upSampleBlock(image, time)
        
        image = self.groupNorm(image)
        image = self.activation(image)
        image = self.outputLayer(image)
        return image
        
        

    


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

    # Define the UNetModel and test it
    unetModel = UNetModel()
    test_image = torch.randn(1, 3, 64, 64)
    test_time = torch.randn(1, 128)
    output_image = unetModel(test_image, test_time)
    print("Output image shape:", output_image.shape)
    print("Done!")