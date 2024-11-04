"""
Training process for the decision transformer model
"""
from typing import Optional

import numpy as np
from rich.progress import track
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model.edt_model import ElasticDecisionTransformer
from .utils.utils import compute_dr
from .utils.edt_utils import expectile_loss
from .utils.return_search import (
    return_search,
    return_search_heuristic
)


class Trainer:
    """
    Decision transformer trainer
    """

    def __init__(
        self,
        year: int,
        model: ElasticDecisionTransformer,
        max_len: int
    ) -> None:
        self.year = year
        self.model = model
        self.max_len = max_len

    def train(
        self,
        epochs: int,
        tr_loader: DataLoader,
    ) -> None:
        """
        trainer

        Training data:
            - state_batch: torch.Tensor with shape (B, T, state_dim)
            - action_batch: torch.Tensor with shape (B, T, action_dim)
            - timesteps_batcj: torch.Tensor with shape (B, T)
            - return_to_go_batch: torch.Tensor with shape (B, T)
            - mask_batch: torch.Tensor with shape (B, T)
        """
        self.model.load_state_dict(torch.load('model_weights/original.pth'))
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in track(range(epochs)):
            total_sample = 0
            total_loss = 0

            for data in tr_loader:
                state_batch, action_batch, timesteps_batch, return_to_go_batch, mask_batch = data

                with torch.autocast(device_type="cuda", dtype=torch.float32):  # automatic mixed precision
                    state_pred, action_pred, return_pred, imp_return_pred, _ = self.model(
                        state_batch.float().cuda(),
                        action_batch.float().cuda(),
                        timesteps_batch.long().cuda(),
                        return_to_go_batch.float().cuda(),
                        attention_mask=mask_batch.bool().cuda()
                    )
                    action_target = torch.clone(action_batch).detach()
                    state_target = torch.clone(state_batch).detach()
                    return_target = torch.clone(return_to_go_batch).detach()
                    action_loss = self.model.loss_fn(action_pred, action_target)
                    state_loss = F.mse_loss(state_pred, state_target, reduction='mean')
                    return_loss = F.cross_entropy(return_pred, return_target)
                    imp_loss = expectile_loss(imp_return_pred, return_target)
                    edt_loss = (
                        action_loss
                        + state_loss
                        + return_loss
                        + imp_loss
                    )

                optimizer.zero_grad()
                edt_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                optimizer.step()
                total_sample += state_batch.shape[0]
                total_loss += edt_loss.item()

            if (epoch + 1) % 10 == 0:
                epoch_loss = total_loss / total_sample
                print(f'Epoch [{epoch+1}/{epochs}] loss = {epoch_loss}')

            if (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(), f'model_weights/{self.year}_len{self.max_len}.pth')

    def test(
        self,
        device: torch.device,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        trg: int = 1,
        heuristic: bool = False,
        top_percentile: float = 0.15,
        expert_weight: float = 10.0,
        mgdt_sampling: bool = False,
        rs_steps: int = 2,
        rs_ratio: float = 1.0,
        real_rtg: bool = False,
        heuristic_delta: float = 1.0,
        previous_index: Optional[int] = None,
    ):
        """
        test the model
        """
        self.model.load_state_dict(torch.load(f'model_weights/{self.year}_len{self.max_len}.pth'))
        self.model.eval()
        dt_weights, indices = [], []
        target_return = torch.Tensor(
            trg,
            device=device,
            dtype=torch.float32
        ).reshape(1, 1)
        rewards_batch = torch.Tensor(
            0,
            device=device,
            dtype=torch.float32
        )
        state_batch = torch.zeros(
            (0, state_size),
            device=device,
            dtype=torch.float32
        )
        action_batch = torch.zeros(
            (1, action_size),
            device=device,
            dtype=torch.float32
        )
        timesteps_batch = torch.Tensor(
            0,
            device=device,
            dtype=torch.long
        ).reshape(1, 1)
        have_position = False
        act = None

        with torch.no_grad():
            print(len(state_test[:-1]))
            for idx, data in track(enumerate(state_test[:-1]), description=f'Test [{self.year}]'):
                state_batch = torch.cat(
                    [state_batch, data.reshape(1, state_size)],
                    dim=0
                )

                if state_batch.shape[0] > self.max_len:
                    state_batch = state_batch[1:]

                if not heuristic:
                    act, best_index = return_search(
                        model=self.model,
                        timesteps=timesteps_batch,
                        states=state_batch,
                        actions=action_batch,
                        rewards_to_go=target_return,
                        rewards=rewards_batch,
                        context_len=self.max_len,
                        t=idx,
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                        mgdt_sampling=mgdt_sampling,
                        rs_steps=rs_steps,
                        rs_ratio=rs_ratio,
                        real_rtg=real_rtg,
                    )
                else:
                    act, best_index = return_search_heuristic(
                        model=self.model,
                        timesteps=timesteps_batch,
                        states=state_batch,
                        actions=action_batch,
                        rewards_to_go=target_return,
                        rewards=rewards_batch,
                        context_len=self.max_len,
                        t=idx,
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                        mgdt_sampling=mgdt_sampling,
                        rs_steps=rs_steps,
                        rs_ratio=rs_ratio,
                        real_rtg=real_rtg,
                        heuristic_delta=heuristic_delta,
                        previous_index=previous_index,
                    )
                    previous_index = best_index

                indices.append(best_index)
                reward_pred, have_position, act = compute_dr(
                    data[-1],
                    state_test[idx + 1, -1],
                    act,
                    have_position,
                    act
                )
                next_target_return = target_return[0, -1] - reward_pred
                target_return = torch.cat([target_return, next_target_return.reshape(1, 1)], dim=1)

                if target_return.shape[1] > self.max_len:
                    target_return = target_return[:, 1:]

                if timesteps_batch.shape[1] < self.max_len:
                    timesteps_batch = torch.cat(
                        [
                            timesteps_batch,
                            torch.ones((1, 1), device=device, dtype=torch.long) * (idx+1)
                        ],
                        dim=1
                    )

        np.save(f'act_weights/dt_weights_{self.year}_len{self.max_len}_{trg}', np.array(dt_weights))
