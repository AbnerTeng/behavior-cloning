import numpy as np
from rich.progress import track
import torch
from torch import optim
import torch.nn.functional as F

from .create_dataset import TradeLogDataset
from .model.rl_model import DecisionTransformer
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
            tr_loader: TradeLogDataset,
    ) -> None:
        """
        trainer
        """
        self.model.load_state_dict(torch.load('./model_weights/original.pth'))
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in range(epochs):
            total_sample = 0
            total_loss = 0

            for data in track(tr_loader, total=len(tr_loader), description=f'Epoch [{epoch+1}/{epochs}]'):
                optimizer.zero_grad()
                state_batch, action_batch, timesteps_batch, return_to_go_batch, mask_batch = data
                _, action_pred, _ = self.model(
                    state_batch,
                    action_batch,
                    return_to_go_batch,
                    timesteps_batch,
                    attention_mask=mask_batch
                )
                loss = self.model.loss_fn(action_pred, action_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                optimizer.step()
                total_sample += state_batch.shape[0]
                total_loss += loss.item()

            epoch_loss = total_loss / total_sample
            print(f'Epoch [{epoch+1}/{epochs}] loss = {epoch_loss}')

            if (epoch+1) % 50 == 0:
                torch.save(self.model.state_dict(), f'./model_weights/{self.year}_len{self.max_len}.pth')

    def test(
            self,
            device: torch.device,
            state_size: int,
            action_size: int,
            state_test: torch.Tensor,
            target_return: int = 1,
    ):
        """
        test the model
        """
        self.model.load_state_dict(torch.load(f'./model_weights/{self.year}_len{self.max_len}.pth'))
        self.model.eval()
        dt_weights = []
        target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        state_batch = torch.zeros((0, state_size), device=device, dtype=torch.float32)
        action_batch = torch.zeros((1, action_size), device=device, dtype=torch.float32)
        timesteps_batch = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        with torch.no_grad():
            for idx, data in enumerate(track(state_test[:-1]), description=f'Test [{self.year}]'):
                state_batch = torch.cat(
                    [
                        state_batch,
                        torch.from_numpy(data).reshape(1, state_size).to(
                            device=device,
                            dtype=torch.float32
                        )
                    ],
                    dim=0
                )

                if state_batch.shape[0] > self.max_len:
                    state_batch = state_batch[1:]

                min_val = torch.min(state_batch, axis=0).values
                max_val = torch.max(state_batch, axis=0).values
                state_norm = (state_batch-min_val) / (max_val-min_val)

                for i, v in enumerate(max_val):
                    if v == min_val[i]:
                        state_norm[:, i] = state_batch[:, i] - min_val[i]

                action_pred = self.model.get_action(state_norm, action_batch, target_return, timesteps_batch)
                action_pred = F.softmax(action_pred)

                if idx >= self.max_len - 1:
                    dt_weights.append(action_pred.tolist())

                action_batch = torch.cat([action_batch, action_pred.reshape(1, action_size)], dim=0)

                if action_batch.shape[0] > self.max_len:
                    action_batch = action_batch[1:]

                reward_pred = compute_dr(data[[0, 3, 6]], state_test[idx+1][[0, 3, 6]], action_pred)
                next_target_return = target_return[0, -1] - reward_pred
                target_return = torch.cat([target_return, next_target_return.reshape(1, 1)], dim=1)

                if target_return.shape[1] > self.max_len:
                    target_return = target_return[:, 1:]

                if timesteps_batch.shape[1] < self.max_len:
                    timesteps_batch = torch.cat(
                        [
                            timesteps_batch,
                            torch.ones((1, 1), device=device, dtype=torch.long) * (idx+1)],
                        dim=1
                    )

        np.save(f'./act_weights/dt_weights_{self.year}_len{self.max_len}_{target_return}', np.array(dt_weights))
