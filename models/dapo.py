import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm

from models.policy_model import PolicyModel
from models.reward_model import RewardModel

class DAPOConfig:
    def __init__(self, grup_size: int=4,
                 eps_low: float=0.2, eps_high: float=0.3,
                 max_length: int=512, length_cache: int=50,
                 learning_rate: float =1e-5, max_grad_norm: float=1.0):
        """Initialize DAPO Config

        Args:
            group_size: Number of outputs to sample per prompt (G).
            eps_low: Lower clipping threshold for PPO.
            eps_high: Upper clipping threshold for PPO.
            max_length: Maximum allowed generation length.
            length_cache: Interval for soft penalty before max_length.
            learning_rate: Learning rate for policy optimization.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        self.group_size = grup_size
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.max_length = max_length
        self.length_cache = length_cache
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

class DAPOAgent:
    def __init__(self, policy_model: PolicyModel,
                 reward_model: RewardModel,
                 config: DAPOConfig):
        """Initialize the DAPO agent.
        
        Args:
            policy_model: Policy model for generating outputs.
            reward_model: Reward model for evaluating outputs.
            config: Configuration for the DAPO algorithm.
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config
        self.device = next(policy_model.parameters().device)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        self.logger = logging.getLogger('dapo')
    def training_step(self, prompts: List[str],
                      prompt_ids: torch.Tensor,
                      prompt_mask: torch.Tensor,
                      answer_keys: Optional[List[Any]] = None) -> Dict[str, float]:
        """Perform one DAPO training step on a batch of prompts.
        
        Args:
            prompts: List of prompts (questions/tasks) to train on.
            prompt_ids: Tensor of prompt token IDs.
            prompt_mask: Tensor of prompt attention mask.
            answer_keys: Optional list of answer keys for prompts.
            
        Returns:
            Dictionary of training metrics.
        """
        config = self.config
        G = config.group_size
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        old_policy =  PolicyModel(self.policy.model.config.name_or_path,
                                  self.policy.tokenizer.name_or_path)
        old_policy.load_state_dict(self.policy.state_dict())
        old_policy.to(self.device)
        old_policy.eval()