from abc import abstractmethod
from typing import Optional, Dict

import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader

from src.trade_env.trade_env import TradeEnv


class BaseTrainer:

    def __init__(
        self,
        year: int,
        model: torch.nn.Module,
        max_len: int,
        expr_name: str,
        model_type: str,
        is_elastic: bool
    ) -> None:
        """
        Args:
            year (int) -> current training / testing year
            model (torch.nn.Module) -> DT model
            max_len (int) -> max sequence length in a batch
        """
        self.year = year
        self.model = model
        self.max_len = max_len
        self.expr_name = expr_name
        self.model_type = model_type
        self.is_elastic = is_elastic

    @abstractmethod
    def dt_train(self,):
        raise NotImplementedError

    @abstractmethod
    def edt_train(self,):
        raise NotImplementedError

    def train(
        self,
        epochs: int,
        tr_loader: DataLoader,
        device: str,
        opt_cfg: Dict[str, float],
        **kwargs
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
        self.model.load_state_dict(
            torch.load(f'ckpts/{self.model_type}/{self.expr_name}_model_weights/original.pth')
        )
        self.model.to(device)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), **opt_cfg)

        for epoch in track(range(epochs)):
            total_sample, total_loss = 0, 0
            
            for data in tr_loader:
                if self.is_elastic:
                    total_loss, total_sample = self.edt_train(
                        data,
                        device,
                        total_sample,
                        total_loss,
                        optimizer,
                        **kwargs
                    )

                else:
                    total_loss, total_sample = self.dt_train(
                        data, device, total_sample, total_loss, optimizer
                    )

            if (epoch + 1) % 10 == 0:
                epoch_loss = total_loss / total_sample
                print(f'Epoch [{epoch+1}/{epochs}] loss = {epoch_loss}')

            if (epoch + 1) % 50 == 0:
                torch.save(
                    self.model.state_dict(),
                    f'ckpts/{self.model_type}/{self.expr_name}_model_weights/{self.year}_len{self.max_len}.pth'
                )

    @abstractmethod
    def dt_test(self):
        raise NotImplementedError

    @abstractmethod
    def edt_test(self):
        raise NotImplementedError


    def test(
        self,
        device: torch.device,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        state_test_denorm: Optional[torch.Tensor] = None,
        rtg_target: int = 1,
        **kwargs
    ) -> None:
        raise NotImplementedError
