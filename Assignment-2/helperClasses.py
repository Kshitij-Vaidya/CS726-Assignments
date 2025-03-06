import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define a class for the positional encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension

    def forward(self, time : torch.Tensor):
        # time : [batchSize]
        device = time.device
        halfDimension = self.embedDimension // 2
        # Create a vector of position
        scale = math.log(10000) / (halfDimension - 1)
        exponents = torch.arange(halfDimension, device=device, dtype=torch.float) * -scale
        exponents = exponents.unsqueeze(0).unsqueeze(0)
        time = time.float().unsqueeze(-1)
        print(time.shape)
        sinuosoidInput = time * exponents
        embeddings = torch.cat([torch.sin(sinuosoidInput), torch.cos(sinuosoidInput)], dim=-1)
        return embeddings


# Define a class for the UNet Module
class UNetBlock(nn.Module):
    def __init__(self, inputChannels, outputChannels, timeEmbedDimensions, numGroups = 8, isTimeEmbedding=True):
        super().__init__()
        self.convLayer1 = nn.Conv2d(inputChannels, outputChannels, kernel_size=3, padding=1)
        self.convLayer2 = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.groupNorm1 = nn.GroupNorm(numGroups, outputChannels)
        self.groupNorm2 = nn.GroupNorm(numGroups, outputChannels)
        self.timeEmbedProj = nn.Linear(timeEmbedDimensions, outputChannels)
        self.activationLayer = nn.SiLU()
        self.isTimeEmbedding = isTimeEmbedding
    
    def forward(self, input, timeEmbed):
        output = self.convLayer1(input)
        output = self.groupNorm1(output)
        output = self.activationLayer(output)
        # Add the time embedding
        if self.isTimeEmbedding:
            output = output + self.timeEmbedProj(timeEmbed)[:, :, None, None]
        output = self.convLayer2(output)
        output = self.groupNorm2(output)
        output = self.activationLayer(output)
        return output
        
# Define a self attention class
class SelfAttention(nn.Module):
    def __init__(self, channels, numGroups=8):
        super().__init__()
        self.groupNorm = nn.GroupNorm(numGroups, channels)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.projOut = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, input):
        B, C, H, W = input.shape
        h = self.groupNorm(input)
        query = self.query(h).reshape(B, C, -1)
        key = self.key(h).reshape(B, C, -1)
        value = self.value(h).reshape(B, C, -1)
        # Compute the attention weights
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = attention / math.sqrt(C)
        attention = F.softmax(attention, dim=-1)
        output = torch.bmm(value, attention.permute(0, 2, 1))
        output = output.reshape(B, C, H, W)
        output = self.projOut(output)
        return input + output # Residual connection



# Define a class for the DDPM model as a UNet Module using the UNetBlock and SelfAttention classes
class UNetModel(nn.Module):
    def __init__(self, inputChannels = 3, 
                 outputChannels = 3, 
                 baseChannels = 64, 
                 timeEmbedDimension = 256):
        super().__init__()
        # Define the downsample path
        self.downsampleBlock1 = UNetBlock(inputChannels, baseChannels, timeEmbedDimension)
        self.downsamplePool1 = nn.MaxPool2d(2)
        self.downsampleBlock2 = UNetBlock(baseChannels, baseChannels * 2, timeEmbedDimension)
        self.downsamplePool2 = nn.MaxPool2d(2)
        self.downsampleBlock3 = UNetBlock(baseChannels * 2, baseChannels * 4, timeEmbedDimension)

        # Add a self attention layer with 16*16 resolution
        self.selfAttention = SelfAttention(baseChannels * 4)
        self.dowmsamplePool3 = nn.MaxPool2d(2)  
        # Bottle neck layer
        self.bottleNeckLayer = UNetBlock(baseChannels * 4, baseChannels * 8, timeEmbedDimension)

        # Define the upsample path
        self.upsampleLayer3 = nn.ConvTranspose2d(baseChannels * 8, baseChannels * 4, kernel_size=2, stride=2)
        self.upsampleBlock3 = UNetBlock(baseChannels * 8, baseChannels * 4, timeEmbedDimension)
        self.upsampleLayer2 = nn.ConvTranspose2d(baseChannels * 4, baseChannels * 2, kernel_size=2, stride=2)
        self.upsampleBlock2 = UNetBlock(baseChannels * 4, baseChannels * 2, timeEmbedDimension)
        self.upsampleLayer1 = nn.ConvTranspose2d(baseChannels * 2, baseChannels, kernel_size=2, stride=2)
        self.upsampleBlock1 = UNetBlock(baseChannels * 2, baseChannels, timeEmbedDimension)

        self.outputConvLayer = nn.Conv2d(baseChannels, outputChannels, kernel_size=1)
    

    def forward(self, input, timeEmbed):
        # Downsample path
        down1 = self.downsampleBlock1(input, timeEmbed)
        pool1 = self.downsamplePool1(down1)
        down2 = self.downsampleBlock2(pool1, timeEmbed)
        pool2 = self.downsamplePool2(down2)
        down3 = self.downsampleBlock3(pool2, timeEmbed)
        # Self attention layer
        down3 = self.selfAttention(down3)
        pool3 = self.dowmsamplePool3(down3)
        # Bottle neck layer
        bottleNeck = self.bottleNeckLayer(pool3, timeEmbed)
        # Upsample path
        up3 = self.upsampleLayer3(bottleNeck)
        up3 = self.upsampleBlock3(torch.cat([down3, up3], dim=1), timeEmbed)
        up2 = self.upsampleLayer2(up3)
        up2 = self.upsampleBlock2(torch.cat([down2, up2], dim=1), timeEmbed)
        up1 = self.upsampleLayer1(up2)
        up1 = self.upsampleBlock1(torch.cat([down1, up1], dim=1), timeEmbed)
        # Output layer
        output = self.outputConvLayer(up1)
        return output
    

if __name__ == '__main__':
    # Test the PositionalEncoding class
    time = torch.arange(200)
    posEnc = PositionalEncoding(256)
    output = posEnc(time)
    print(output.shape)
