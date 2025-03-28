import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generateTokens = []
        currentInput = input_ids

        for _  in range(self.max_output_len):
            # Get the output logits
            output = self.model(currentInput)
            logits = output.logits[:, -1, :] # Take the last token's logits
            # Select the token with the highest probability
            nextToken = torch.argmax(logits, dim=-1)
            # Stop if the token is EOS
            if nextToken.item() == self.eos_token_id:
                break
            # Append the token to the generated tokens
            generateTokens.append(nextToken.item())
            # Update the input for the next iteration
            currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
        return torch.tensor(generateTokens, dtype=torch.long)
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generatedTokens = []
        currentInput = input_ids

        for _ in range(self.max_output_len):
            # Get the output model logits
            output = self.model(currentInput)
            logits = output.logits[:, -1, :] # Take the last token's logits
            # Apply temperature to the logits
            probabilities = nn.functional.softmax(logits / self.tau, dim=-1)
            # Sample a token from the distribution
            nextToken = torch.multinomial(probabilities, num_samples=1)
            # Stop if the token is EOS
            if nextToken.item() == self.eos_token_id:
                break
            # Append the token to the generated tokens
            generatedTokens.append(nextToken.item())
            # Update the input for the next iteration
            currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
        return torch.tensor(generatedTokens, dtype=torch.long)
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generatedTokens = []
        currentInput = input_ids

        for _ in range(self.max_output_len):
            # Get the output model logits
            output = self.model(currentInput)
            logits = output.logits[:, -1, :]
            # Select the top-k tokens
            topkLogits, topkIndices = torch.topk(logits, self.k, dim=-1)
            probabilities = nn.functional.softmax(topkLogits.values / self.tau, dim=-1)
            # Sample a token from the distribution
            nextToken = topkIndices[0, torch.multinomial(probabilities, num_samples=1)]
            # Stop if the token is EOS
            if nextToken.item() == self.eos_token_id:
                break
            # Append the token to the generated tokens
            generatedTokens.append(nextToken.item())
            # Update the input for the next iteration
            currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
        return torch.tensor(generatedTokens, dtype=torch.long)
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        generatedTokens = []
        currentInput = input_ids

        for _ in range(self.max_output_len):
            # Get the output model logits
            output = self.model(currentInput)
            logits = output.logits[:, -1, :]
            # Sort the logits and compute the cumulative distribution
            sortedLogits, sortedIndices = torch.sort(logits, descending=True, dim=-1)
            probabilities = nn.functional.softmax(sortedLogits / self.tau, dim=-1)
            cumulativeProbabilities = torch.cumsum(probabilities, dim=-1)
            # Keep only the tokens within the nucleus (top-p)
            nucleusMask = cumulativeProbabilities <= self.p
            nucleusLogits = sortedLogits[nucleusMask]
            nucleusIndices = sortedIndices[nucleusMask]
            # Sample a token from the distribution
            nucleusProbabilities = nn.functional.softmax(nucleusLogits, dim=-1)
            nextToken = nucleusIndices[torch.multinomial(nucleusProbabilities, num_samples=1)]

            # Stop if the token is EOS
            if nextToken.item() == self.eos_token_id:
                break
            # Append the token to the generated tokens
            generatedTokens.append(nextToken.item())
            # Update the input for the next iteration
            currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
        return torch.tensor(generatedTokens, dtype=torch.long) 