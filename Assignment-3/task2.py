import evaluate
import torch 
import numpy as np
import random
import argparse
import time

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from medusa.model.medusa_model_new import MedusaModel
from task0 import set_seed, get_dataloader, clean_text

# -------------------------------
# New Implementation of MedusaTextGenerator
# -------------------------------
class MedusaTextGenerator:
    def __init__(self, model, use_no_medusa_heads, decoding_strategy, eos_id, beam_width, max_output_len):
        """
        Parameters:
          model: The Medusa model with one LM head and multiple medusa heads.
          use_no_medusa_heads (S): Number of medusa heads to be used.
          decoding_strategy: "single-head" for greedy or "multi-head" for beam search speculative decoding.
          eos_id: The end-of-sequence token id.
          beam_width (W): Beam width for multi-head decoding.
          max_output_len: Maximum total length of generated tokens.
        """
        self.model = model
        self.use_no_medusa_heads = use_no_medusa_heads  # S (number of medusa heads)
        self.decoding_strategy = decoding_strategy
        self.eos_id = eos_id
        self.beam_width = beam_width
        self.max_output_len = max_output_len

    def __call__(self, input_ids):
        """
        Generate tokens given input_ids.
        Depending on decoding_strategy, uses single-head (greedy) or multi-head (beam search) decoding.
        """
        # Ensure batch size is 1 for simplicity.
        generated = input_ids.clone()
        while generated.shape[1] < self.max_output_len:
            if self.decoding_strategy == "single-head":
                # --- Single Head (Greedy) Decoding ---
                outputs = self.model(generated)  # Expecting outputs[0] from LM head
                lm_logits = outputs[0]  # shape: [1, seq_len, vocab_size]
                # Take logits for the last token
                next_token_logits = lm_logits[:, -1, :]  # [1, vocab_size]
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self.eos_id:
                    break

            elif self.decoding_strategy == "multi-head":
                # --- Multi Head (Medusa Speculative Decoding) ---
                # Step 1: Run the model once to get S+1 distributions:
                # Assume model returns a tuple where:
                # outputs[0] is LM head logits for token at position t,
                # outputs[1] ... outputs[S] are medusa head logits for tokens at t+1 ... t+S.
                outputs = self.model(generated)
                # Save the current length (t) for clarity.
                t = generated.shape[1]
                S = self.use_no_medusa_heads  # number of medusa heads
                # Prepare beam search: start with current context.
                candidates = [(generated, 0.0)]
                # For each of the S+1 future tokens:
                for s in range(S + 1):
                    # Get the appropriate head's logits.
                    # s == 0 corresponds to LM head; s>=1 to medusa heads.
                    head_logits = outputs[s]  # shape: [1, t, vocab_size] (note: t is current sequence length)
                    # We assume that for each head, the prediction is for a single token.
                    # So we take the logits corresponding to the last position.
                    logits = head_logits[:, -1, :]  # shape: [1, vocab_size]
                    # Compute log probabilities
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # shape: [vocab_size]

                    new_candidates = []
                    # Extend each candidate in the beam
                    for cand, score in candidates:
                        # Get top beam_width tokens from current head
                        topk = torch.topk(log_probs, self.beam_width)
                        for token, token_log_prob in zip(topk.indices, topk.values):
                            # Extend candidate by appending this token
                            new_candidate = torch.cat([cand, token.unsqueeze(0)], dim=1)
                            new_score = score + token_log_prob.item()
                            new_candidates.append((new_candidate, new_score))
                    # Keep only the top beam_width candidates
                    new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
                    candidates = new_candidates

                # Step 3: Re-score each candidate extension using LM head.
                # We want to compute the LM-head score for the newly added S+1 tokens.
                best_score = -float('inf')
                best_candidate = None
                # original context length is t
                for cand, _ in candidates:
                    # We extract the extension tokens (positions t to t+S)
                    extension = cand[0, t:]
                    # Re-run LM head on the full candidate sequence.
                    lm_out = self.model(cand)[0]  # LM head logits for cand, shape: [1, seq_len, vocab_size]
                    # Sum the log probabilities for the extension tokens.
                    # We only compute for positions corresponding to the extension.
                    ext_score = 0.0
                    for i, token in enumerate(extension, start=t):
                        token_logits = lm_out[:, i, :]
                        token_log_probs = torch.log_softmax(token_logits, dim=-1)
                        ext_score += token_log_probs[0, token].item()
                    if ext_score > best_score:
                        best_score = ext_score
                        best_candidate = cand

                # Update generated with the extension from the best candidate.
                extension = best_candidate[0, t:]
                generated = best_candidate

                # Check if the extension contains EOS.
                if self.eos_id in extension.tolist():
                    break

            else:
                raise ValueError("Unknown decoding strategy provided.")

        # Return generated tokens (squeeze batch dimension)
        return generated[0]

# -------------------------------
# Main evaluation script in task2.py
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name","-m", type=str, required=False, default="FasterDecoding/medusa-v1.0-vicuna-7b-v1.5", help="The Huggingface medusa model to be used for inference")
    parser.add_argument("--hf-token","-token", type=str, required=True, help="The Huggingface token for accessing Llama weights")
    parser.add_argument("--use-no-medusa-heads","-nmh", type=int, required=False, default=2, help="The number of medusa heads to be used for inference")
    parser.add_argument("--max-input-len","-mil", type=int, required=False, default=1000, help="Maximum length of the input sequence.")
    parser.add_argument("--max-output-len","-mol", type=int, required=False, default=50, help="Maximum number of new tokens to be generated.")
    parser.add_argument("--beam-width","-w", type=int, required=False, default=2, help="Size of beam width for beam search.")
    parser.add_argument("--decoding-strategy","-ds", type=str, required=False, default="single-head", choices=["single-head", "multi-head"], help="The decoding strategy to be used during inference.")
    parser.add_argument("--debug","-db",type=bool,default=False, help="To print debugging statements.")
    
    args = parser.parse_args() 
    print(args)
    
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the Medusa model with FP16 precision and move it to device.
    model = MedusaModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = model.get_tokenizer()
    model.eval()
    
    # Use our new MedusaTextGenerator implementation.
    generator = MedusaTextGenerator(
        model=model,
        use_no_medusa_heads=args.use_no_medusa_heads,
        decoding_strategy=args.decoding_strategy,
        eos_id=tokenizer.eos_token_id,
        beam_width=args.beam_width,
        max_output_len=args.max_output_len
    )
    
    # Load dataset
    dataloader = get_dataloader(tokenizer, args.hf_token, max_input_len=args.max_input_len)
    
    reference_texts = []
    generated_texts = []
    
    total = len(dataloader)
    total_time = 0
    total_generated_tokens = 0
  
    for i, batch in enumerate(dataloader):
        input_prompt, ground_truth = batch 
        
        # Process the reference text: tokenize, truncate, decode and clean.
        reference_text = [[tokenizer.decode(tokenizer.encode(out)[:args.max_output_len], skip_special_tokens=True, clean_up_tokenization_spaces=True)] for out in ground_truth][0][0]
        reference_text = clean_text(reference_text)
        reference_texts.append(reference_text)
        
        token_ids = input_prompt['input_ids'].to(device)
        
        start_time = time.time()
        generated_tokens = generator(token_ids)
        end_time = time.time()
        
        generated_text = tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_texts.append(generated_text)
        
        total_time += (end_time - start_time)
        total_generated_tokens += len(generated_tokens)
        
        if args.debug:
            print(f'Example: {i+1}/{total}')
            print(f'Input Prompt:', tokenizer.decode(input_prompt['input_ids'][0]))
            print('Reference:', reference_texts[-1])
            print('Generated:', generated_texts[-1])
            print()
            print()
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    
    bleu_score = bleu.compute(predictions=generated_texts, references=reference_texts, max_order=1)
    rouge_score = rouge.compute(predictions=generated_texts, references=reference_texts)
    
    rtf = total_time / total_generated_tokens if total_generated_tokens > 0 else float('inf')
    
    print(f"""BLEU: {bleu_score['bleu']}
ROUGE-1: {float(rouge_score['rouge1'])}
ROUGE-2: {float(rouge_score['rouge2'])}
ROUGE-LCS: {float(rouge_score['rougeL'])}
RTF:{rtf}""")
