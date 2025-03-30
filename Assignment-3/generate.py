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
        max_output_len: int = 100,
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
        print("[TASK 0] Decoder Called ...")
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
        past_key_values = None

        for i in range(self.max_output_len):
            # if i % 20 == 0:
                # print(f"[TASK 0] Greedy Decoding Stage {i} ...")
            # Get the output logits
            with torch.no_grad():
                output = self.model(currentInput, past_key_values = past_key_values)
                # print("[TASK 0] Model Output Received ...")
                logits = output.logits[:, -1, :] # Take the last token's logits
                past_key_values = output.past_key_values if hasattr(output, "past_key_values") else None
            # Select the token with the highest probability
            nextToken = torch.argmax(logits, dim=-1)
            # Stop if the token is EOS
            if nextToken.item() == self.eos_token_id:
                print("[DEBUG] EOS token detected ... Breaking from the loop")
                break
            # Append the token to the generated tokens
            generateTokens.append(nextToken.item())
            # Update the input for the next iteration
            currentInput = nextToken.unsqueeze(0).to(input_ids.device)
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
        past_key_values = None

        with torch.no_grad():
            for i in range(self.max_output_len):
                # print(f"[DEBUG] Processing {i} ... ")
                # Get the output model logits
                output = self.model(currentInput, past_key_values = past_key_values)
                logits = output.logits[:, -1, :] # Take the last token's logits
                past_key_values = output.past_key_values if hasattr(output, "past_key_values") else None
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
                if past_key_values is None:
                    currentInput = torch.cat([currentInput, nextToken], dim=1)
                else:
                    currentInput = nextToken

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
        past_key_values = None

        with torch.no_grad():
            for i in range(self.max_output_len):
                # print(f"[DEBUG] Processing {i} ... ")
                # Get the output model logits
                output = self.model(currentInput, past_key_values = past_key_values)
                logits = output.logits[:, -1, :]
                past_key_values = output.past_key_values if hasattr(output, "past_key_values") else None
                # Select the top-k tokens
                topkLogits, topkIndices = torch.topk(logits, self.k, dim=-1)
                probabilities = nn.functional.softmax(topkLogits / self.tau, dim=-1)
                # Sample a token from the distribution
                nextToken = topkIndices[0, torch.multinomial(probabilities, num_samples=1).squeeze(1)]
                # Stop if the token is EOS
                if nextToken.item() == self.eos_token_id:
                    break
                # Append the token to the generated tokens
                generatedTokens.append(nextToken.item())
                # Update the input for the next iteration
                if past_key_values is None:
                    currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
                else:
                    currentInput = nextToken.unsqueeze(0)
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
        past_key_values = None 

        for i in range(self.max_output_len):
            # Forward pass through the model
            with torch.no_grad():
                output = self.model(currentInput, past_key_values = past_key_values)
                logits = output.logits[:, -1, :]
                past_key_values = output.past_key_values if hasattr(output, "past_key_values") else None 
            logits = logits / self.tau
            probabilities = nn.functional.softmax(logits, dim=-1).squeeze()
            sortedProbabilities, sortedIndices = torch.sort(probabilities, descending=True)

            cumProbs = torch.cumsum(sortedProbabilities, dim=0)
            nucleusMask = cumProbs <= self.p

            if not torch.any(nucleusMask):
                nucleusMask[0] = True 

            # Get the indices of the selected tokens
            selectedIndices = sortedIndices[nucleusMask]
            selectedProbabilities = sortedProbabilities[nucleusMask]
            # Renormalise the probabilites
            print(f"[DEBUG] Probability Sum : {torch.sum(selectedProbabilities)}")
            selectedProbabilities = selectedProbabilities / torch.sum(selectedProbabilities)

            # Sample from the selected tokens
            nextToken = torch.multinomial(selectedProbabilities, num_samples=1)
            nextToken = selectedIndices[nextToken]

            generatedTokens.append(nextToken.item())

            if nextToken.item() == self.eos_token_id:
                break

            if past_key_values is None:
                currentInput = torch.cat([currentInput, nextToken.unsqueeze(0)], dim=-1)
            else:
                currentInput = nextToken.unsqueeze(0)
        return torch.tensor(generatedTokens, dtype=torch.long)

