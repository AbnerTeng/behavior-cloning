from typing import Any, Dict, List, Union
import yaml
import numpy as np
import torch


def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    Load configuration file
    """
    with open(cfg_path, 'r', encoding="utf-8") as cfg:
        config = yaml.safe_load(cfg)

    return config


def min_max_norm(x: np.ndarray) -> np.ndarray:
    """
    min-max normalization for 3-dim matrix

    keep the shape of min, max as (n, 20, 1)
    """
    normed_array = np.zeros_like(x)

    for strat in range(x.shape[0]):
        min_val = np.min(x[strat], axis=1, keepdims=True)
        max_val = np.max(x[strat], axis=1, keepdims=True)
        normed_array[strat] = (x[strat] - min_val) / (max_val - min_val)

    return normed_array


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
    close_tomorrow: float,
    action_today: np.ndarray,
    have_position: bool,
    prev_act: int | None,
    trans_cost: float = 0.001,
    portfolio_value: float = 1000000
) -> float:
    """
    compute daily return

    input:
    - n: amount of strategies

    - close_today: float
    - close_tomorrow: float
    - action_today: (4,) vector [buy, sell, buytocover, short]
    """
    action = np.argmax(action_today.cpu().numpy())

    if isinstance(close_today, torch.Tensor):
        close_today = close_today.cpu().numpy()

    if close_today == 0.0:
        close_today += 1e-4

    share_today = np.floor(portfolio_value / close_today)
    cash_today = portfolio_value - share_today * close_today
    trans_cost_today = share_today * close_today * trans_cost
    portfolio_value_tommorow = share_today * close_tomorrow + cash_today - trans_cost_today
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

        return dr, have_position, prev_act

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


def get_slicev2(
    data: Union[np.ndarray, torch.Tensor],
    num_strats: int,
    max_len: int = 20,
    start: int = 0,
    end: int | None = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    slice the data into length of training

    The training length -> first day until the specifc year (end - max_len)
    """
    if end is not None:
        return data[(start * num_strats): (end * num_strats), :]

    return data[(start * num_strats):, :]


def get_test_slicev2(
    data: np.ndarray,
    num_strats: int,
    train_len: int,
    max_len: int,
    year_start_idx: int,
    test_year: int,
    year_list: List[str],
) -> np.ndarray:
    """
    slice the data into length of testing

    The testing length -> last 19 days of training data until the specific year end
    """
    start = train_len - (max_len - 1)  # start from last 19 days of training data
    end = None if test_year == year_list[-1] else year_start_idx[test_year + 1]
    return get_slicev2(data, num_strats, 0, start, end)
