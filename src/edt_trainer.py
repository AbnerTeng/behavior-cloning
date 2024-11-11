"""
Training process for the decision transformer model
"""
import os

import numpy as np
from rich.progress import track
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model.edt_model import ElasticDecisionTransformer
from .utils.edt_utils import expectile_loss
from .utils.return_search import (
    return_search,
    # return_search_heuristic
)
from .env.trade_env import TradeEnv


class EDTTrainer:
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
        device: torch.device,
        expr_name: str,
        expectile: float = 0.99,
        state_loss_weight: float = 1.0,
        exp_loss_weight: float = 0.5,
        ce_weight: float = 0.001
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
        self.model.load_state_dict(torch.load(f'{expr_name}_model_weights/original.pth'))
        self.model.to(device)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in track(range(epochs)):  # epochs = max_train_iters
            total_sample = 0
            total_loss = 0

            for data in tr_loader:  # len(tr_loader) = num_updates per iter
                state_batch, _, action_batch, timesteps_batch, return_to_go_batch, mask_batch = data
                return_to_go_batch = return_to_go_batch.unsqueeze(-1)

                with torch.autocast(device_type="cuda", dtype=torch.float32):  # automatic mixed precision
                    state_pred, action_pred, return_pred, imp_return_pred, _ = self.model(
                        state_batch.float().to(device),
                        action_batch.float().to(device),
                        timesteps_batch.long().to(device),
                        return_to_go_batch.float().to(device),
                        attention_mask=mask_batch.bool().to(device)
                    )
                    action_target = torch.clone(action_batch).detach()
                    state_target = torch.clone(state_batch).detach()
                    return_target = torch.clone(return_to_go_batch).detach()
                    action_loss = self.model.loss_fn(action_pred, action_target)
                    state_loss = F.mse_loss(state_pred, state_target, reduction='mean')
                    return_loss = F.cross_entropy(return_pred, return_target)
                    imp_loss = expectile_loss(
                        (imp_return_pred - return_target),
                        expectile=expectile
                    ).mean()
                    edt_loss = (
                        action_loss
                        + state_loss * state_loss_weight
                        + imp_loss * exp_loss_weight
                        + return_loss * ce_weight
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
                torch.save(self.model.state_dict(), f'{expr_name}_model_weights/{self.year}_len{self.max_len}.pth')

    def test(
        self,
        device: torch.device,
        expr_name: str,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        rtg_target: int = 1,
        # heuristic: bool = False,
        top_percentile: float = 0.15,
        expert_weight: float = 10.0,
        mgdt_sampling: bool = False,
        rs_steps: int = 2,
        rs_ratio: float = 1.0,
        real_rtg: bool = False,
        # heuristic_delta: float = 1.0,
        # previous_index: Optional[int] = None,
    ):
        """
        test the model

        state_test.shape -> shape(290, 4) at year 2011
        state_size: (int) -> 4
        action_size: (int) -> 4
        trg: (int) -> 1
        """
        env = TradeEnv(state_test)
        eval_batch_size = 1
        num_eval_ep = 1
        total_reward = 0
        indices, edt_weights = [], []

        timesteps = torch.arange(
            start=0, end=state_test.shape[0] + 2 * self.max_len, step=1
        )
        timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
        self.model.load_state_dict(torch.load(f'{expr_name}_model_weights/{self.year}_len{self.max_len}.pth'))
        self.model.eval()

        with torch.no_grad():
            for _ in range(num_eval_ep):
                actions = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, action_size),
                    dtype=torch.float32,
                    device=device
                )
                states = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, state_size),
                    dtype=torch.float32,
                    device=device
                )
                rewards_to_go = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, 1),
                    dtype=torch.float32,
                    device=device
                )
                rewards = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, 1),
                    dtype=torch.float32,
                    device=device
                )

                # initialize episode
                initial_state = env.reset()  # assume it's a (20, 4) matrix
                running_state = initial_state[:19, :]
                running_reward = 0
                running_rtg = rtg_target

                for t in track(range(state_test.shape[0] - self.max_len), description=f'Test [{self.year}]'):
                    states[0, t: t + 19] = running_state
                    running_rtg -= running_reward
                    rewards_to_go[0, t] = running_rtg
                    rewards[0, t] = running_reward

                    # if not heuristic:
                    action_pred, best_index = return_search(
                        model=self.model,
                        timesteps=timesteps,
                        states=states,
                        actions=actions,
                        rewards_to_go=rewards_to_go,
                        context_len=self.max_len,
                        t=t,
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                        mgdt_sampling=mgdt_sampling,
                        rs_steps=rs_steps,
                        rs_ratio=rs_ratio,
                        real_rtg=real_rtg,
                    )
                    indices.append(best_index)
                    # else:
                    #     act, best_index = return_search_heuristic(
                    #         model=self.model,
                    #         timesteps=timesteps_batch,
                    #         states=state_batch,
                    #         actions=action_batch,
                    #         rewards_to_go=target_return,
                    #         rewards=rewards_batch,
                    #         context_len=self.max_len,
                    #         t=idx,
                    #         top_percentile=top_percentile,
                    #         expert_weight=expert_weight,
                    #         mgdt_sampling=mgdt_sampling,
                    #         rs_steps=rs_steps,
                    #         rs_ratio=rs_ratio,
                    #         real_rtg=real_rtg,
                    #         heuristic_delta=heuristic_delta,
                    #         previous_index=previous_index,
                    #     )
                    #     previous_index = best_index

                    # indices.append(best_index)

                    new_state, running_reward, done, _ = env.step(np.argmax(action_pred).item())
                    last_state = new_state[-1:, :]
                    running_state = torch.cat(
                        [running_state[1:], last_state],
                        dim=0
                    )
                    print(action_pred)
                    actions[0, t] = action_pred
                    total_reward += running_reward
                    edt_weights.append(action_pred.tolist())

                    if done:
                        break

        if f'{expr_name}_act_weights' not in os.listdir():
            os.mkdir(f'{expr_name}_act_weights')

        np.save(
            f'{expr_name}_act_weights/edt_weights_{self.year}_len{self.max_len}_{rtg_target}',
            np.array(edt_weights)
        )
