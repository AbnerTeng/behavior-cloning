from abc import abstractmethod
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DecisionTransformerConfig, DecisionTransformerGPT2Model


class BaseDecisionTransformer(nn.Module):
    """
    Base class of decision transformer module
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        config = DecisionTransformerConfig(
            state_dim=self.state_size,
            act_dim=self.action_size,
            hidden_size=self.hidden_size,
            max_ep_len=4096,
        )
        self.transformer = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(self.state_size, self.hidden_size)
        self.embed_rtg = nn.Linear(1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        return_to_go: torch.Tensor,
        attention_mask=None,
    ) -> Tuple:
        """
        forward pass
        """
        raise NotImplementedError

    def loss_fn(self, action_pred: torch.Tensor, action_target: torch.Tensor) -> None:
        """
        Loss function

        input:
        - action_pred: (n, d, action_size)
        - action_target: (n, d, action_size)

        Ouptut: Loss value
        """
        if action_target.device != action_pred.device:
            action_target = action_target.to(action_pred.device)

        action_pred = F.softmax(action_pred, dim=-1)
        action_pred = action_pred.permute(0, 2, 1)
        action_target = action_target.permute(0, 2, 1)
        ce_loss = nn.CrossEntropyLoss(reduction="sum")
        loss = ce_loss(action_pred, action_target)

        return loss

    @abstractmethod
    def get_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Union[float, Tuple[float, torch.Tensor, torch.Tensor]]:
        """
        get predicted action
        """
        raise NotImplementedError
