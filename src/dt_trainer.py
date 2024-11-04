"""
Training process for the decision transformer model
"""
import numpy as np
from rich.progress import track
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model.dt_model import DecisionTransformer
from .utils.utils import compute_dr

class Trainer:
    """
    Decision transformer trainer
    """
    def __init__(
        self,
        year: int,
        model: DecisionTransformer,
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
        """
        self.model.load_state_dict(torch.load('model_weights/original.pth'))
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in track(range(epochs)):
            total_sample = 0
            total_loss = 0

            for data in tr_loader:
                optimizer.zero_grad()
                state_batch, action_batch, timesteps_batch, return_to_go_batch, mask_batch = data
                _, _, action_pred = self.model(
                    state_batch.float().cuda(),
                    action_batch.float().cuda(),
                    timesteps_batch.long().cuda(),
                    return_to_go_batch.float().cuda(),
                    attention_mask=mask_batch.bool().cuda()
                )
                action_target = torch.clone(action_batch).detach()
                loss = self.model.loss_fn(action_pred, action_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                optimizer.step()
                total_sample += state_batch.shape[0]
                total_loss += loss.item()

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
    ):
        """
        test the model
        """
        self.model.load_state_dict(torch.load(f'model_weights/{self.year}_len{self.max_len}.pth'))
        self.model.eval()
        dt_weights = []
        target_return = torch.tensor(
            trg,
            device=device,
            dtype=torch.float32
        ).reshape(1, 1)
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
        timesteps_batch = torch.tensor(
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

                action_pred = self.model.get_action(
                    state_batch,
                    action_batch,
                    target_return,
                    timesteps_batch
                )
                action_pred = F.softmax(action_pred)

                if torch.isnan(action_pred).any() is True:
                    print(f'Nan detected at {idx}')
                    print(action_batch)
                    print(target_return)
                    raise ValueError('Nan detected')

                if idx >= self.max_len - 1:
                    dt_weights.append(action_pred.tolist())

                action_batch = torch.cat([action_batch, action_pred.reshape(1, action_size)], dim=0)

                if action_batch.shape[0] > self.max_len:
                    action_batch = action_batch[1:]

                reward_pred, have_position, act = compute_dr(
                    data[-1],
                    state_test[idx + 1, -1],
                    action_pred,
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
