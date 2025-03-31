import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from medusa.model.medusa_model_new import MedusaModel
from typing import List, Optional

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
        tokenizer: Optional = None,
    ) -> None:
        self.model = model
        self.decoding_strategy = decoding_strategy
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        self.tokenizer = tokenizer

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
        return self.generator_func(input_ids)

    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        generated = input_ids.clone()
        while generated.shape[1] < self.max_output_len:
            outputs = self.model(generated)
            lm_logits = outputs.logits
            next_token_logits = lm_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            if next_token.item() == self.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=1)
        return generated[0, input_ids.shape[1]:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        S = self.no_heads - 1  
        generated = input_ids.clone()

        while generated.shape[1] < self.max_output_len:
            t = generated.shape[1]
            candidates = [(generated, 0.0)]
            for s in range(S + 1):
                new_candidates = []
                for cand, score in candidates:
                    outputs_candidate = self.model(cand)
                    if not isinstance(outputs_candidate, (list, tuple)):
                        outputs_candidate = [outputs_candidate] * self.no_heads
                    elif len(outputs_candidate) < self.no_heads:
                        outputs_candidate = list(outputs_candidate) + [outputs_candidate[0]] * (self.no_heads - len(outputs_candidate))

                    head_output = outputs_candidate[s]
                    head_logits = head_output.logits
                    logits = head_logits[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                    topk = torch.topk(log_probs, self.beam_width)
                    for token, token_log_prob in zip(topk.indices, topk.values):
                        token = token.unsqueeze(0).unsqueeze(0)
                        new_cand = torch.cat([cand, token], dim=1)
                        new_score = score + token_log_prob.item()
                        new_candidates.append((new_cand, new_score))
                candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]

            best_score = -float('inf')
            best_candidate = None
            for cand, _ in candidates:
                extension = cand[0, t:]
                lm_out = self.model(cand)[0]
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
                break
        return generated[0, input_ids.shape[1]:]



