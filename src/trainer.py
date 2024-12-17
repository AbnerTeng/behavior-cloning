"""
Training process for the decision transformer model
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track

from .base.base_trainer import BaseTrainer
from .trade_env.trade_env import TradeEnv
from .utils.return_search import return_search, return_search_heuristic
from .utils.utils import compute_dr, expectile_loss


class Trainer(BaseTrainer):
    """
    Decision transformer trainer
    """

    def dt_train(
        self,
        data: torch.Tensor,
        device: str,
        total_sample: int,
        total_loss: float,
    ) -> Tuple[float, int]:
        (
            state_batch,
            _,
            action_batch,
            timesteps_batch,
            return_to_go_batch,
            mask_batch,
        ) = data

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            _, _, action_pred = self.model(
                state_batch.float().to(device),
                action_batch.float().to(device),
                timesteps_batch.long().to(device),
                return_to_go_batch.float().to(device),
                attention_mask=mask_batch.bool().to(device),
            )

        action_target = torch.clone(action_batch).detach()
        loss = self.model.loss_fn(action_pred, action_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        total_sample += state_batch.shape[0]
        total_loss += loss.item()

        return total_loss, total_sample

    def edt_train(
        self,
        data: torch.Tensor,
        device: str,
        total_sample: int,
        total_loss: float,
        expectile: float = 0.99,
        state_loss_weight: float = 1.0,
        exp_loss_weight: float = 0.5,
        ce_weight: float = 0.001,
    ) -> Tuple[float, int]:
        (
            state_batch,
            _,
            action_batch,
            timesteps_batch,
            return_to_go_batch,
            mask_batch,
        ) = data
        return_to_go_batch = return_to_go_batch.unsqueeze(-1)

        with torch.autocast(
            device_type="cuda", dtype=torch.float32
        ):  # automatic mixed precision
            state_pred, action_pred, return_pred, imp_return_pred, _ = self.model(
                state_batch.float().to(device),
                action_batch.float().to(device),
                timesteps_batch.long().to(device),
                return_to_go_batch.float().to(device),
                attention_mask=mask_batch.bool().to(device),
            )
            action_target = torch.clone(action_batch).detach()
            state_target = torch.clone(state_batch).detach()
            return_target = torch.clone(return_to_go_batch).detach()
            action_loss = self.model.loss_fn(action_pred, action_target)
            state_loss = F.mse_loss(state_pred, state_target, reduction="mean")
            return_loss = F.cross_entropy(return_pred, return_target)
            imp_loss = expectile_loss(
                (imp_return_pred - return_target), expectile=expectile
            ).mean()
            edt_loss = (
                action_loss
                + state_loss * state_loss_weight
                + imp_loss * exp_loss_weight
                + return_loss * ce_weight
            )

        self.optimizer.zero_grad()
        edt_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        total_sample += state_batch.shape[0]
        total_loss += edt_loss.item()

        return total_loss, total_sample

    # TODO: refactor test functions of dt & edt
    def dt_test(
        self,
        device: str,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        trg: int = 1,
    ) -> None:
        """
        test the model
        """
        self.model.load_state_dict(
            torch.load(
                f"{self.folder_name}/{self.expr_name}_model_weights/{self.year}_len{self.max_len}.pth"
            )
        )
        self.model.eval()
        dt_weights = []
        target_return = torch.tensor(trg, device=device, dtype=torch.float32).reshape(
            1, 1
        )
        state_batch = torch.zeros((0, state_size), device=device, dtype=torch.float32)
        action_batch = torch.zeros((1, action_size), device=device, dtype=torch.float32)
        timesteps_batch = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        have_position = False
        act = None

        with torch.no_grad():
            print(len(state_test[:-1]))
            for idx, data in track(
                enumerate(state_test[:-1]), description=f"Test [{self.year}]"
            ):
                state_batch = torch.cat(
                    [state_batch, data.reshape(1, state_size)], dim=0
                )

                if state_batch.shape[0] > self.max_len:
                    state_batch = state_batch[1:]

                action_pred = self.model.get_action(
                    state_batch, action_batch, target_return, timesteps_batch
                )
                action_pred = F.softmax(action_pred)

                if torch.isnan(action_pred).any() is True:
                    print(f"Nan detected at {idx}")
                    print(action_batch)
                    print(target_return)
                    raise ValueError("Nan detected")

                if idx >= self.max_len - 1:
                    dt_weights.append(action_pred.tolist())

                action_batch = torch.cat(
                    [action_batch, action_pred.reshape(1, action_size)], dim=0
                )

                if action_batch.shape[0] > self.max_len:
                    action_batch = action_batch[1:]

                reward_pred, have_position, act = compute_dr(
                    data[-1], state_test[idx + 1, -1], action_pred, have_position, act
                )
                next_target_return = target_return[0, -1] - reward_pred
                target_return = torch.cat(
                    [target_return, next_target_return.reshape(1, 1)], dim=1
                )

                if target_return.shape[1] > self.max_len:
                    target_return = target_return[:, 1:]

                if timesteps_batch.shape[1] < self.max_len:
                    timesteps_batch = torch.cat(
                        [
                            timesteps_batch,
                            torch.ones((1, 1), device=device, dtype=torch.long)
                            * (idx + 1),
                        ],
                        dim=1,
                    )

        if f"{self.expr_name}_act_weights" not in os.listdir(f"{self.folder_name}"):
            os.mkdir(f"{self.folder_name}/{self.expr_name}_act_weights")

        np.save(
            f"{self.folder_name}/{self.expr_name}_act_weights/dt_weights_{self.year}_len{self.max_len}_{trg}",
            np.array(dt_weights),
        )

    def edt_test(
        self,
        device: str,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        state_test_denorm: torch.Tensor,
        rtg_target: int = 1,
        heuristic: bool = False,
        top_percentile: float = 0.15,
        expert_weight: float = 10.0,
        mgdt_sampling: bool = False,
        rs_steps: int = 2,
        rs_ratio: float = 1.0,
        real_rtg: bool = False,
        heuristic_delta: float = 1.0,
        previous_index: Optional[int] = None,
    ) -> None:
        """
        test the model

        state_test.shape -> shape(290, 4) at year 2011
        state_test_denorm.shape -> shape(290, 4) at year 2011
        state_size: (int) -> 4
        action_size: (int) -> 4
        trg: (int) -> 1
        """
        env = TradeEnv(state_test, state_test_denorm)
        eval_batch_size = 1
        num_eval_ep = 1
        total_reward = 0
        indices, edt_weights = [], []

        timesteps = torch.arange(
            start=0, end=state_test.shape[0] + 2 * self.max_len, step=1
        )
        timesteps = timesteps.repeat(eval_batch_size, 1).to(device)
        self.model.load_state_dict(
            torch.load(
                f"{self.folder_name}/{self.expr_name}_model_weights/{self.year}_len{self.max_len}.pth"
            )
        )
        self.model.eval()

        with torch.no_grad():
            for _ in range(num_eval_ep):
                actions = torch.zeros(
                    (
                        eval_batch_size,
                        state_test.shape[0] + 2 * self.max_len,
                        action_size,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                states = torch.zeros(
                    (
                        eval_batch_size,
                        state_test.shape[0] + 2 * self.max_len,
                        state_size,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                rewards_to_go = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, 1),
                    dtype=torch.float32,
                    device=device,
                )
                rewards = torch.zeros(
                    (eval_batch_size, state_test.shape[0] + 2 * self.max_len, 1),
                    dtype=torch.float32,
                    device=device,
                )

                # initialize episode
                initial_state = env.reset()  # assume it's a (20, 4 / 5) matrix
                running_state = initial_state
                running_reward = 0
                running_rtg = rtg_target

                for t in track(
                    range(state_test.shape[0] - self.max_len),
                    description=f"Test [{self.year}]",
                ):
                    states[0, t : t + 20] = running_state
                    running_rtg -= running_reward
                    rewards_to_go[0, t] = running_rtg
                    rewards[0, t] = running_reward

                    if not heuristic:
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

                    else:
                        action_pred, best_index = return_search_heuristic(
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
                            heuristic_delta=int(heuristic_delta),
                            previous_index=previous_index,
                        )
                        previous_index = best_index
                        indices.append(best_index)

                    new_state, running_reward, done, _ = env.step(
                        np.argmax(action_pred.cpu()).item()
                    )
                    last_state = new_state[-1:, :]
                    running_state = torch.cat(
                        [running_state[1:], torch.unsqueeze(last_state, 0)], dim=0
                    )
                    actions[0, t] = F.softmax(action_pred)
                    total_reward += running_reward
                    edt_weights.append(action_pred.tolist())

                    if done:
                        break

        if heuristic:
            act_file_name = f"{self.expr_name}_act_weights_hs"
        else:
            act_file_name = f"{self.expr_name}_act_weights"

        if act_file_name not in os.listdir(f"{self.folder_name}"):
            os.mkdir(f"{self.folder_name}/{act_file_name}")

        np.save(
            f"{self.folder_name}/{act_file_name}/edt_weights_{self.year}_len{self.max_len}_{rtg_target}",
            np.array(edt_weights),
        )
