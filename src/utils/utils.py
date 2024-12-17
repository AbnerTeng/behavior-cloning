import os
from typing import Dict, Tuple, List, Union, Optional

from omegaconf import ListConfig, OmegaConf, DictConfig
import numpy as np
import torch


def load_config(cfg_path: str) -> Union[DictConfig, ListConfig]:
    """
    Load configuration file
    """
    config = OmegaConf.load(cfg_path)

    return config


def set_all_seed(seed: int) -> None:
    """
    Set seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_num_files(log_path: str) -> int:
    """
    Count logs inside a main subject
    """
    try:
        count_files = len(os.listdir(f"trade_log/{log_path}"))

    except FileNotFoundError:
        count_files = 0

    return count_files


def binary_transfer(act: torch.Tensor) -> torch.Tensor:
    """
    Squeeze the binary action into 1-dim

    Example: [n, m, 4] -> [n, m]

    [0, 0, 0, 1] -> 1
    [0, 1, 0, 0] -> 3
    [0, 0, 0, 0] -> 0
    """
    action_space = [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
    ]

    act = torch.tensor(act, dtype=torch.int64)
    action_to_index = {action: idx for idx, action in enumerate(action_space)}
    indices = torch.tensor(
        [[action_to_index[tuple(a.tolist())] for a in sequence] for sequence in act]
    )

    return indices


def min_max_norm(x: np.ndarray, state_with_vol: bool) -> np.ndarray:
    """
    min-max normalization for 3-dim matrix

    keep the shape of min, max as (n, 20, 1)
    """
    if state_with_vol:
        vol = x[:, :, -1]
        x = x[:, :, :-1]

    vol = None
    normed_array = np.zeros_like(x)

    for strat in range(x.shape[0]):
        min_val = np.min(x[strat], axis=1, keepdims=True)
        max_val = np.max(x[strat], axis=1, keepdims=True)
        normed_array[strat] = (x[strat] - min_val) / (max_val - min_val)

    if state_with_vol and vol is not None:
        vol = np.expand_dims(vol, axis=-1)
        concat_array = np.concatenate([normed_array, vol], axis=-1)
        return concat_array
    else:
        return normed_array


def expectile_loss(diff: torch.Tensor, expectile: float = 0.99) -> torch.Tensor:
    """
    diff (torch.Tensor): difference between the target and the imp_loss_return
    expectile (float): expectile value (default: 0.99)
    """
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def count_return_to_go(returns: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the return to go (sum of future rewards)
    """
    rtg = np.zeros_like(returns)

    for strat in range(returns.shape[0]):
        rtg[strat][-1] = returns[strat][-1]

        for t in reversed(range(returns[strat].shape[0] - 1)):
            rtg[strat][t] = returns[strat][t] + gamma * rtg[strat][t + 1]

    return rtg


def compute_dr(
    close_today,
    close_tomorrow: torch.Tensor,
    action_today: torch.Tensor,
    have_position: bool,
    prev_act: int | None,
    trans_cost: float = 0.001,
    portfolio_value: float = 1000000,
) -> Tuple[float, bool, Optional[int]]:
    """
    compute daily return

    input:
    - n: amount of strategies

    - close_today: float
    - close_tomorrow: float
    - action_today: (4,) vector [buy, sell, buytocover, short]
    """
    action = int(np.argmax(action_today.cpu().numpy()))
    dr, prev_act = 0, None

    if isinstance(close_today, torch.Tensor):
        close_today = close_today.cpu().numpy()

    if close_today == 0.0:
        close_today += 1e-4

    share_today = np.floor(portfolio_value / close_today)
    cash_today = portfolio_value - share_today * close_today
    trans_cost_today = share_today * close_today * trans_cost
    portfolio_value_tommorow = (
        share_today * close_tomorrow + cash_today - trans_cost_today
    )
    long_dr = (portfolio_value_tommorow - portfolio_value) / portfolio_value
    short_dr = (-portfolio_value_tommorow + portfolio_value) / portfolio_value

    if have_position is False:
        if action == 0:
            have_position = True
            dr = long_dr

        if action == 3:
            have_position = True
            dr = short_dr

        else:
            dr = 0

        prev_act = action

    if have_position is True:
        if prev_act == 0:
            if action in [0, 2]:
                have_position = True
                prev_act = 0
                dr = long_dr

            if action in [1, 3]:
                have_position = False
                prev_act = action
                dr = 0

        elif prev_act == 3:
            if action in [1, 3]:
                have_position = True
                prev_act = 3
                dr = short_dr

            if action in [0, 2]:
                have_position = False
                prev_act = action
                dr = 0

    return dr, have_position, prev_act


def get_start_year_idx(year_list: List[int], timesteps: np.ndarray) -> Dict[int, int]:
    """
    Get the start index of each year
    """
    year_start_idx = {}

    for y in year_list:
        for idx, date in enumerate(timesteps[-1]):
            if str(y) in date:
                year_start_idx[y] = idx
                break

    return year_start_idx


def get_slicev2(
    data: List[torch.Tensor],
    num_strats: int,
    start: int = 0,
    end: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    slice the data into length of training

    The training length -> first day until the specifc year (end - max_len)
    """
    output = []

    for d in data:
        if end is not None:
            output.append(d[(start * num_strats) : (end * num_strats), :])
        else:
            output.append(d[(start * num_strats) :, :])

    return output


def get_test_slicev2(
    data: List[torch.Tensor],
    num_strats: int,
    train_len: int,
    max_len: int,
    year_start_idx: Dict[int, int],
    test_year: int,
    year_list: List[int],
) -> List[torch.Tensor]:
    """
    slice the data into length of testing

    The testing length -> last 19 days of training data until the specific year end
    """
    start = train_len - (max_len - 1)  # start from last 19 days of training data
    end = None if test_year == year_list[-1] else year_start_idx[test_year + 1]

    return get_slicev2(data, num_strats, start, end)
