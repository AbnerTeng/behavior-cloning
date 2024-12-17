from abc import abstractmethod
from typing import Optional, Tuple, Any

import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader


class BaseTrainer:
    """
    Args:
        year (int) -> current training / testing year
        model (torch.nn.Module) -> DT model
        max_len (int) -> max sequence length in a batch
    """

    def __init__(
        self,
        year: int,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        max_len: int,
        folder_name: str,
        expr_name: str,
        model_type: str,
        is_elastic: bool,
    ) -> None:
        self.year = year
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_len = max_len
        self.folder_name = folder_name
        self.expr_name = expr_name
        self.model_type = model_type
        self.is_elastic = is_elastic

    @abstractmethod
    def dt_train(
        self,
        data: torch.Tensor,
        device: str,
        total_sample: int,
        total_loss: float,
    ):
        raise NotImplementedError

    @abstractmethod
    def edt_train(
        self,
        data: torch.Tensor,
        device: str,
        total_sample: int,
        total_loss: float,
        expectile: float,
        state_loss_weight: float,
        exp_loss_weight: float,
        ce_weight: float,
    ) -> Tuple[float, int]:
        raise NotImplementedError

    def train(
        self,
        epochs: int,
        tr_loader: DataLoader,
        device: str,
        **kwargs,
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
            torch.load(
                f"{self.folder_name}/{self.expr_name}_model_weights/original.pth"
            )
        )
        self.model.to(device)
        self.model.train()

        for epoch in track(range(epochs)):
            total_sample, total_loss = 0, 0

            for data in tr_loader:
                if self.is_elastic:
                    total_loss, total_sample = self.edt_train(
                        data, device, total_sample, total_loss, **kwargs
                    )

                else:
                    total_loss, total_sample = self.dt_train(
                        data,
                        device,
                        total_sample,
                        total_loss,
                    )

                if self.scheduler is not None:
                    self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                epoch_loss = total_loss / total_sample
                print(f"Epoch [{epoch+1}/{epochs}] loss = {epoch_loss}")

            if (epoch + 1) % 50 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.folder_name}/{self.expr_name}_model_weights/{self.year}_len{self.max_len}.pth",
                )

    @abstractmethod
    def dt_test(
        self,
        device: str,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        trg: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def edt_test(
        self,
        device: str,
        state_size: int,
        action_size: int,
        state_test: torch.Tensor,
        state_test_denorm: torch.Tensor,
        rtg_target: int,
        heuristic: bool,
        top_percentile: float,
        expert_weight: float,
        mgdt_sampling: bool,
        rs_steps: int,
        rs_ratio: float,
        real_rtg: bool,
        heuristic_delta: float,
        previous_index: Optional[int],
    ) -> None:
        raise NotImplementedError

    def test(
        self,
        # device: str,
        # state_size: int,
        # action_size: int,
        # state_test: torch.Tensor,
        # state_test_denorm: Optional[torch.Tensor] = None,
        # rtg_target: int = 1,
        # **kwargs,
    ) -> None:
        raise NotImplementedError
