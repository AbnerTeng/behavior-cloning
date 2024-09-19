"""
Decision transformer model modified from minyu
"""
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerGPT2Model
)


class DecisionTransformer(nn.Module):
    """
    Decision Transformer model
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_size: int,
            max_len: int
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        config = DecisionTransformerConfig(
            state_dim = self.state_size,
            act_dim = self.action_size,
            hidden_size = hidden_size,
            max_ep_len = max_len
        )
        self.transformer = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(self.state_size, self.hidden_size)
        self.embed_action = nn.Linear(self.action_size, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_size)
        self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, config.act_dim),
            nn.Tanh()
            #nn.Softmax(dim=-1)
        )

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            return_to_go: torch.Tensor,
            timesteps: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        forward pass
        """
        batch_size, seq_length = state.shape[0], state.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(state.device)

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(state) + time_embeddings
        action_embeddings = self.embed_action(action) + time_embeddings
        returns_embeddings = self.embed_return(return_to_go) + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings),
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask),
            dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask = stacked_attention_mask
        )

        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_pred = self.predict_return(x[:, 2])
        state_pred = self.predict_state(x[:, 2])
        action_pred = self.predict_action(x[:, 1])

        return state_pred, action_pred, return_pred

    def loss_fn(self, action_pred: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
        """
        loss function
        """
        action_pred = action_pred.reshape(-1, self.action_size)
        action_pred = F.softmax(action_pred, dim=1)
        action_target = action_target.reshape(-1, self.action_size)
        loss = torch.mean((action_pred - action_target) ** 2)

        return loss

    def get_action(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            return_to_go: torch.Tensor,
            timesteps: torch.Tensor
    ) -> float:
        """
        get action
        """
        state = state.reshape(1, -1, self.state_size)
        action = action.reshape(1, -1, self.action_size)
        return_to_go = return_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        attention_mask = None

        _, action_pred, _ = self.forward(state, action, return_to_go, timesteps, attention_mask=attention_mask)

        return action_pred[0, -1]
