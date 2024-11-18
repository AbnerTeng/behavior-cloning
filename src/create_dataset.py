"""
Dataset creation module
"""
from typing import Tuple, Optional

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


class EDTTradeLogDataset(Dataset):
    """
    Dataset of Elastic Decision Transformer trading logs
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

    Args:
        max_len: int --> window size for the sequence
        state_size: int --> state size (4 here)
        action_size: int --> action size (4 here)
        gamma: float --> discount factor for return to go

    """
    def __init__(
        self,
        max_len: int,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
    ) -> None:
        self.max_length = max_len
        self.max_ep_len = 4096
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def split_data(
        self,
        state_set: np.ndarray,
        next_state_set: Optional[np.ndarray],
        action_set: np.ndarray,
        return_set: np.ndarray,
    ) -> Tuple[torch.Tensor]:
        """
        Split the full data into subsets based on the length of the sequence (self.max_len)

        Number of iterations: (state.shape[1] - self.max_length + 1)
        in each iter, we slice a piece of the data with window size = self.max_length

        Data format: stack row wise with every 20 days of data

        Args:
            state_set: np.ndarray --> full states with shape (k, d, 4)
            next_state_set: np.ndarray --> full next states with shape (k, d, 4)
            action_set: np.ndarray --> full actions with shape (k, d, 4)
            return_set: np.ndarray --> full returns with shape (k, d)

        Returns:
            Tuple[torch.Tensor]: (s, a, rt, t, mask)
            s --> stacked state with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length, 4)
            norm_s --> stacked normalized state with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length, 4)
            ns --> stacked next state with shape
                (k * (next_state_set.shape[1] - self.max_length+1)), self.max_length, 4)
            a --> stacked action with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length, 4)
            rt --> stacked return to go with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length)
            t --> stacked timesteps with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length)
                output of every t: [0, 1, ..., 19]
            mask --> stacked mask with shape
                (k * (state_set.shape[1] - self.max_length+1)), self.max_length)
        """
        s, norm_s, ns, a, t, rt, mask = [], [], [], [], [], [], []

        for start_idx in range(state_set.shape[1] - self.max_length + 1):
            state_piece = state_set[:, start_idx: start_idx + self.max_length, :]  # (k, 20, 4)
            seq_length = state_piece.shape[1]
            state_norm = min_max_norm(state_piece)
            s.append(state_piece)
            norm_s.append(state_norm)

            if next_state_set is not None:
                next_state_piece = next_state_set[:, start_idx: start_idx + self.max_length, :]  # (k, 20, 4)
                seq_length = next_state_piece.shape[1]
                next_state_norm = min_max_norm(next_state_piece)
                ns.append(next_state_norm)

            action_piece = action_set[:, start_idx: start_idx + self.max_length, :]  # (k, 20, 4)
            a.append(action_piece)
            timesteps = np.arange(self.max_length)
            expand_timesteps = np.tile(timesteps, state_set.shape[0]).reshape(state_set.shape[0], timesteps.shape[0])
            t.append(expand_timesteps)
            returns_to_go_piece = count_return_to_go(
                return_set[:, start_idx: start_idx + self.max_length],  # (k, 20)
                self.gamma
            )
            rt.append(returns_to_go_piece)
            mask.append(np.ones((state_set.shape[0], seq_length)))

        s = torch.from_numpy(
            np.concatenate(s, axis=0)
        ).to(dtype=torch.float32, device=self.device)

        norm_s = torch.from_numpy(
            np.concatenate(norm_s, axis=0)
        ).to(dtype=torch.float32, device=self.device)

        if next_state_set is not None:
            ns = torch.from_numpy(
                np.concatenate(ns, axis=0)
            ).to(dtype=torch.float32, device=self.device)

        a = torch.from_numpy(
            np.concatenate(a, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        t = torch.from_numpy(
            np.concatenate(t, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        rtg = torch.from_numpy(
            np.concatenate(rt, axis=0)
        ).to(dtype=torch.float32, device=self.device)
        mask = torch.from_numpy(
            np.concatenate(mask, axis=0)
        ).to(dtype=torch.float32, device=self.device)

        return s, norm_s, ns, a, t, rtg, mask
