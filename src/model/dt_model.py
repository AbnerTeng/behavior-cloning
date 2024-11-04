## src/model/rl_model.py

"""
Decision transformer model modified from minyu
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
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
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        config = DecisionTransformerConfig(
            state_dim = self.state_size,
            act_dim = self.action_size,
            hidden_size = self.hidden_size,
            max_ep_len = 4096
        )
        self.transformer = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(self.state_size, self.hidden_size)
        self.embed_action = nn.Linear(self.action_size, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_state = nn.Linear(self.hidden_size, self.state_size)
        self.predict_return = nn.Linear(self.hidden_size, 1)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_size),
            # nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        return_to_go: torch.Tensor,
        attention_mask = None
    ) -> Tuple:
        """
        forward pass
        """
        batch_size, seq_length, _ = state.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(state.device)

        # time_embeddings = self.embed_timestep(timesteps.squeeze(-1))
        time_embeddings = self.embed_timestep(timesteps)  # (batch_size, seq_length, hidden_size)
        state_embeddings = self.embed_state(state) + time_embeddings  # (batch_size, seq_length, hidden_size)
        action_embeddings = self.embed_action(action) + time_embeddings  # (batch_size, seq_length, hidden_size)
        returns_embeddings = self.embed_return(return_to_go.unsqueeze(-1)) + time_embeddings  # (batch_size, seq_length, hidden_size)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings),
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)  # (batch_size, 3 * seq_length, hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask),
            dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask = stacked_attention_mask
        )

        x = transformer_outputs['last_hidden_state']  # (batch_size, 3 * seq_length, hidden_size)
        x = x.reshape(
            batch_size, seq_length, 3, self.hidden_size
        ).permute(0, 2, 1, 3)  # (batch_size, 3, seq_length, hidden_size))
        return_pred = self.predict_return(x[:, 2])
        state_pred = self.predict_state(x[:, 2])
        action_pred = self.predict_action(x[:, 1])

        return return_pred, state_pred, action_pred

    def loss_fn(
        self,
        action_pred: torch.Tensor,
        action_target: torch.Tensor
    ) -> torch.Tensor:
        """
        loss function

        input:
        - action_pred: (n, d, action_size)
        - action_target: (n, d, action_size)
        """
        action_pred = F.softmax(action_pred, dim=-1)
        action_pred = action_pred.permute(0, 2, 1)
        action_target = action_target.permute(0, 2, 1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(action_pred, action_target)

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

        input:
        - d: length of the sequence

        - state: (d, state_size)
        - action: (1, action_size)
        - timesteps: (1, 1)
        - return_to_go: (1, 1)

        output:
        - action_pred: (n, d, action_size)
        """
        state = state.reshape(1, -1, self.state_size)
        action = action.reshape(1, -1, self.action_size)
        timesteps = timesteps.reshape(1, -1)
        # return_to_go = return_to_go.reshape(1, -1)
        attention_mask = None
        _, _, action_pred = self.forward(
            state,
            action,
            timesteps,
            return_to_go,
            attention_mask=attention_mask
        )

        return action_pred[0, -1]
