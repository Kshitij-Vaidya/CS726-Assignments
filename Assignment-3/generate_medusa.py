import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from medusa.model.medusa_model_new import MedusaModel
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: MedusaModel, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM (MedusaModel)
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        # no_heads = S + 1: the LM head plus S medusa heads.
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        generated = input_ids.clone()
        while generated.shape[1] < self.max_output_len:
            outputs = self.model(generated)
            # Since we are in single-head mode, we expect a single output object.
            lm_logits = outputs.logits  # shape: [1, seq_len, vocab_size]
            next_token_logits = lm_logits[:, -1, :]  # shape: [1, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # shape: [1,1]
            if next_token.item() == self.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=1)
        # Return only the tokens generated beyond the input prompt.
        return generated[0, input_ids.shape[1]:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        # S: number of medusa heads (excluding the LM head)
        S = self.no_heads - 1  
        generated = input_ids.clone()
        debug_mode = True  # Enable debug prints
        
        while generated.shape[1] < self.max_output_len:
            outputs = self.model(generated)
            # If the output is not a tuple, replicate it for all heads.
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs] * self.no_heads
            elif len(outputs) < self.no_heads:
                # If outputs is a tuple with insufficient heads, replicate the LM head.
                outputs = list(outputs) + [outputs[0]] * (self.no_heads - len(outputs))
                
            t = generated.shape[1]
            # Debug: Print logits for each head.
            for s in range(self.no_heads):
                head_output = outputs[s]
                print(f"[DEBUG] Head {s} logits shape: {head_output.logits.shape}")
                print(f"[DEBUG] Head {s} logits sample: {head_output.logits[0, -1, :5].detach().cpu().tolist()}")
            
            # Initialize beam with the current context and score 0.
            candidates = [(generated, 0.0)]
            # For each future token position (simulate S+1 tokens in total)
            for s in range(S + 1):
                new_candidates = []
                head_output = outputs[s]  # For s=0: LM head; for s>=1: Medusa head for token t+s.
                head_logits = head_output.logits  # shape: [1, seq_len, vocab_size]
                logits = head_logits[:, -1, :]     # shape: [1, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # shape: [vocab_size]
                for cand, score in candidates:
                    topk = torch.topk(log_probs, self.beam_width)
                    for token, token_log_prob in zip(topk.indices, topk.values):
                        token = token.unsqueeze(0).unsqueeze(0)  # reshape to [1,1]
                        new_cand = torch.cat([cand, token], dim=1)
                        new_candidates.append((new_cand, score + token_log_prob.item()))
                # Retain only the top beam_width candidates.
                candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
                # Debug print for this beam search step.
                for idx, (cand, score) in enumerate(candidates):
                    last_token = cand[0, -1].item()
                    print(f"[DEBUG] Beam step {s+1}, candidate {idx+1}: last token = {last_token}, cumulative score = {score}")
            
            # Re-score each candidate extension using the LM head.
            best_score = -float('inf')
            best_candidate = None
            for cand, _ in candidates:
                extension = cand[0, t:]
                lm_out = self.model(cand)[0]  # LM head logits for candidate.
                ext_score = 0.0
                for i, token in enumerate(extension, start=t):
                    token_logits = lm_out[:, i, :]
                    token_log_probs = torch.log_softmax(token_logits, dim=-1)
                    ext_score += token_log_probs[0, token].item()
                if ext_score > best_score:
                    best_score = ext_score
                    best_candidate = cand
            new_tokens = best_candidate[0, t:]
            generated = best_candidate
            if self.eos_token_id in new_tokens.tolist():
                print("[DEBUG] EOS token found in multi-head extension. Stopping generation.")
                break
        return generated[0, input_ids.shape[1]:]
