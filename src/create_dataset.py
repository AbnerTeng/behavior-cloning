"""
Dataset creation module
"""
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.utils import (
    min_max_norm,
    count_return_to_go
)


class TradeLogDataset(Dataset):
    """
    Dataset of trading logs
    """

    def __init__(self, *args) -> None:
        self.data = args

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(self, idx: int) -> Tuple:
        return tuple(
            d[idx] for d in self.data
        )


class DataPreprocess:
    """
    Preprocess class
    """
    def __init__(
            self,
            max_len: int,
            state_size: int,
            action_size: int,
            gamma: float,
    ) -> None:
        self.max_length = max_len
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def split_data(
            self,
            state_subset,
            action_subset,
            return_subset
    ) -> Tuple:
        """
        Split the full data into subsets
        """
        s, a, t, rt, mask = [], [], [], [], []

        for start_idx in range(len(state_subset) - (self.max_length - 1)):
            state_piece = state_subset[
                start_idx: start_idx + self.max_length
            ].reshape(1, -1, self.state_size)
            seq_length = state_piece.shape[1]
            state_norm = min_max_norm(state_piece)
            s.append(state_norm)

            action_piece = action_subset[
                start_idx: start_idx + self.max_length
            ].reshape(1, -1, self.action_size)
            a.append(action_piece)

            timesteps = np.arange(self.max_length).reshape(-1, 1)

            returns_to_go_piece = count_return_to_go(
                return_subset[start_idx: start_idx + self.max_length],
                self.gamma
            ).reshape(1, -1, 1)
            rt.append(returns_to_go_piece)

            mask.append(np.ones((1, seq_length)))

        s = torch.from_numpy(
            np.concatenate(s, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(
            np.concatenate(a, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        rtg = torch.from_numpy(
            np.concatenate(rt, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        t = torch.from_numpy(
            np.concatenate(timesteps, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        mask = torch.from_numpy(
            np.concatenate(mask, axis=0)
        ).to(dtype=torch.float32, device=self.device)

        return s, a, rtg, t, mask
