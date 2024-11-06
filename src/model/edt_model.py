from typing import Tuple

import torch
from torch import nn
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerGPT2Model
)

from .dt_model import DecisionTransformer


class ElasticDecisionTransformer(DecisionTransformer):
    """
    Elastic decision transformer model
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        num_inputs: int,
        num_bin: int,
        is_continuous: bool = True
    ) -> None:
        super().__init__(
            state_size,
            action_size,
            hidden_size
        )
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.is_continuous = is_continuous

        config = DecisionTransformerConfig(
            state_dim=self.state_size,
            act_dim=self.action_size,
            hidden_size=self.hidden_size,
            max_ep_len=4096
        )
        self.transformer = DecisionTransformerGPT2Model(
            config
        )

        # projection heads
        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(self.state_size, self.hidden_size)
        self.embed_rtg = nn.Linear(1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # discrete actions
        if not self.is_continuous:
            self.embed_action = nn.Embedding(self.action_size, self.hidden_size)

        else:
            self.embed_action = nn.Linear(self.action_size, self.hidden_size)

        # prediction heads
        self.predict_state = nn.Linear(self.hidden_size + self.action_size, self.state_size)
        self.predict_rtg = nn.Linear(self.hidden_size, int(num_bin))
        self.predict_rtg2 = nn.Linear(self.hidden_size, 1)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(self.hidden_size, self.action_size)]
                + ([nn.Tanh()] if self.is_continuous else [])
            )
        )
        self.predict_reward = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        return_to_go: torch.Tensor,
        attention_mask=None
    ) -> Tuple:
        batch_size, seq_length, _ = state.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(state.device)

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(state) + time_embeddings
        action_embeddings = self.embed_action(action) + time_embeddings
        returns_embeddings = self.embed_rtg(return_to_go.unsqueeze(-1)) + time_embeddings

        stacked_inputs = torch.stack(
            (
                returns_embeddings,
                state_embeddings,
                action_embeddings
            ),
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.num_inputs * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stack_attention_mask = torch.stack(
            [attention_mask] * 3,
            dim=1
        ).reshape(batch_size, 3 * seq_length)
        x = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stack_attention_mask
        )['last_hidden_state']
        x = x.reshape(
            batch_size, seq_length, self.num_inputs, self.hidden_size
        ).permute(0, 2, 1, 3)

        return_pred = self.predict_rtg(x[:, 0])  # predict next return given state
        return_pred2 = self.predict_rtg2(x[:, 0])  # predict next return with implicit loss
        action_pred = self.predict_action(x[:, 1])  # predict action given S, R
        state_pred = self.predict_state(
            torch.cat((x[:, 1], action_pred), 2)
        )
        reward_pred = self.predict_reward(
            x[:, 2]
        )  # predict reward given S, R, A

        return (
            state_pred,
            action_pred,
            return_pred,
            return_pred2,
            reward_pred
        )

    def get_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
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
        _, action_pred, ret_pred, imp_ret_pred, _ = self.forward(
            state,
            action,
            timesteps,
            return_to_go,
            attention_mask=attention_mask
        )

        return action_pred[0, -1], ret_pred.squeeze(-1), imp_ret_pred.squeeze(-1)
