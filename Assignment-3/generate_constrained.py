import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from collections import defaultdict

warnings.filterwarnings("ignore")

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

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
        trie = Trie(self.tokenizer)
        for word in word_list:
            trie.insert(word)
        
        generatedTokens = []
        pastKeyValues = None
        currentInput = input_ids

        for _ in range(self.max_output_len):
            outputs = self.model(currentInput,
                                 past_key_values=pastKeyValues)
            logits = outputs.logits[:, -1, :]
            pastKeyValues = outputs.past_key_values

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            sortedTokens = torch.argsort(probabilities, descending=True)
            nextToken = None

            for token in sortedTokens.squeeze(0):
                token = token.item()
                if trie.startsWith(generatedTokens + [token]):
                    nextToken = token
                    break
            if nextToken is None or nextToken == self.eos_token_id:
                break
                
            generatedTokens.append(nextToken)
            currentInput = torch.tensor([[nextToken]]).to(input_ids.device)
        
        return torch.tensor(generatedTokens, dtype=torch.long)
        


# Implement a Trie Class to handle the word list
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isEndOfWord = False
    

class Trie:
    def __init__(self, tokenizer):
        self.root = TrieNode()
        self.tokenizer = tokenizer
    
    def insert(self, word):
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        node = self.root
        for token in tokens:
            node = node.children[token]
        node.isEndOfWord = True
    
    def startsWith(self, prefixTokens):
        node = self.root
        for token in prefixTokens:
            if token not in node.children:
                return False
            node = node.children[token]
        return True

    def validCompletion(self, prefixTokens) -> Bool:
        node = self.root
        for token in prefixTokens:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.isEndOfWord