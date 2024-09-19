from typing import Any, Dict
import yaml
import numpy as np


def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    Load configuration file
    """
    with open(cfg_path, 'r', encoding="utf-8") as cfg:
        config = yaml.safe_load(cfg)

    return config


def min_max_norm(x: Any) -> np.ndarray:
    """
    min-max normalization
    """
    return (x - np.min(x, axis=1)) / (np.max(x, axis=1) - np.min(x, axis=1))


def count_return_to_go(returns: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the return to go
    """
    return np.flip(np.flip(returns).cumsum()) * gamma ** np.arange(len(returns))


def compute_dr(
        close_today: np.ndarray,
        close_tomorrow: np.ndarray,
        weight_today: np.ndarray,
        trans_cost: float = 0.001,
        portfolio_value: float = 1000000
) -> float:
    """
    compute daily return
    """
    share_today = np.floor(weight_today.cpu().numpy() * portfolio_value / close_today)
    pv_without_cash = np.sum(share_today * close_today)
    cash_today = portfolio_value - pv_without_cash
    trans_cost_today = pv_without_cash * trans_cost
    portfolio_value_tomorrow = sum(share_today * close_tomorrow) + cash_today - trans_cost_today
    return (portfolio_value_tomorrow - portfolio_value) / portfolio_value
