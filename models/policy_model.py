import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class PolicyModel(nn.Module):
    def __init__(self, model_name: str,
                  tokenizer_name: str):
        """Initialize the policy model.
        
        Args:
            model_name: Name of the pretrained model.
            tokenizer_name: Name of the tokenizer.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Model outputs.
        """
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        return outputs.logits
    def generate(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_length: int=512, num_return_sequences: int=1,
                 temperature: float=1.0, top_p: float=1.0,
                 do_sample: bool=True) -> Tuple[torch.Tensor, List[str]]:
        """Generate responses from the policy model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            max_length: Maximum generation length.
            num_return_sequences: Number of sequences to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling (vs greedy search).
            
        Returns:
            Tuple of generated token IDs and decoded text responses.
        """
        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_length': max_length,
            'num_return_sequences': num_return_sequences,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        responses = []
        for i in range(output_ids.shape[0]):
            input_len = input_ids.shape[1]
            response_ids = output_ids[i, input_len:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response_text)
        return output_ids, responses
    def get_token_logprobs(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for each token in the sequence.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            
        Returns:
            Log probabilities tensor of shape (batch_size, sequence_length).
        """
        logits = self(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_input_ids = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        batch_size, seq_len, vocab_size = log_probs.shape
        token_log_probs = log_probs.gather(
            -1, shift_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * shift_mask
        return token_log_probs
    def save(self, save_dir: str) -> None:
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
    @classmethod
    def load(cls, load_dir: str) -> 'PolicyModel':
        model_name = load_dir
        tokenizer_name = load_dir
        instance = cls(model_name, tokenizer_name)
        return instance