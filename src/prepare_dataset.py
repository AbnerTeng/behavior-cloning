"""
self-generated dataset preparation code
"""
import os
import warnings
from typing import Dict, List, Tuple
import pickle

from rich.progress import track
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class PrepareDataset:
    r"""
    Formatting dataset from self-generated data to decision transformer output
    self-generated data format:
    data: {
        "date": List[str],
        "return": List[float],
        "trade_log": List[Dict[str, List[int]]],
        "trajectory": List[Tuple],
        "param": List[object]
    }

    Decision transformer input format (per trajectory)
    - action: np.array \in R^{day, 4} (long, sell, buytocover, short)
    - state: np.array \in R^{day, 4} (open, high, low, close)
        - Volume will be considered in the future
    - return: np.array \in R^{day}
    - timestep: np.ndarray \in R^{day}
    """
    def __init__(
        self,
        generate_data_dir: str,
        state_data_dir: str
    ) -> None:
        self.generate_data_dir = generate_data_dir
        self.state_data_dir = state_data_dir
        self.state_data = pd.read_csv(self.state_data_dir)
        self.action_types = ['long', 'sell', 'buytocover', 'short']
        self.random_log = os.listdir(self.generate_data_dir)[0]
        self.trajectories = {}

    def load_expert_log(self, log: str) -> Tuple[List[Dict[str, List[int]]], List[float], List[str]]:
        """
        loading expert log from self-generated data
        """
        with open(f"{self.generate_data_dir}/{log}", 'rb') as f:
            data = pickle.load(f)
            action = data['trade_log']
            returns = data['return']
            trade_timestep = data['date']

        return action, returns, trade_timestep

    def store_action(self, action: List[Dict[str, List[int]]]) -> np.array:
        """
        Store action data to numpy array
        """
        action_array = np.zeros((len(self.state_data), 4))

        for period in action:
            for idx, name in enumerate(self.action_types):
                if period[name] is not None:
                    action_array[period[name], idx] = 1
                else:
                    continue

        return action_array

    def store_return(self, returns: List[float]) -> np.ndarray:
        return_array = np.zeros(len(self.state_data))
        return_array[len(self.state_data) - len(returns): ] = returns
        return return_array

    def padding(self, trajectory: Dict[str, np.array]) -> Dict[str, np.ndarray]:
        """
        pad the trajectory to recorded timestep
        """
        raise NotImplementedError

    def run(self, state_with_vol: bool, get_next_state: bool, k: int = 17) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run the dataset preparation code, and get the top k return strategies
        """
        strats_type = list(set(log.split("_")[0] for log in os.listdir(self.generate_data_dir)))
        strats_type = [x for x in strats_type if "pkl" not in x]

        for log in track(os.listdir(self.generate_data_dir)):
            action, returns, _ = self.load_expert_log(log)
            action_array = self.store_action(action)
            return_array = self.store_return(returns)
            log_type = log.split("_")[0]

            if "pkl" in log_type:
                log_type = log_type.split(".")[0]

            self.trajectories[log] = {
                "action": action_array,
                "state": self.state_data.drop(
                    columns=['Date', 'Adj Close', 'Volume'] if not state_with_vol else ['Date', 'Adj Close']
                ).to_numpy(),
                "return": return_array,
                "timestep": self.state_data['Date'].to_numpy()
            }

            if get_next_state:
                self.trajectories[log]["next_state"] = self.state_data.drop(
                    columns=['Date', 'Adj Close', 'Volume'] if not state_with_vol else ['Date', 'Adj Close']
                ).shift(periods=-1).to_numpy()

        cum_rets = {key: value['return'].cumsum()[-1] for key, value in self.trajectories.items()}
        cum_rets = sorted(cum_rets.items(), key=lambda x: x[1], reverse=True)[:k]
        self.trajectories = {key: value for key, value in self.trajectories.items() if key in [x[0] for x in cum_rets]}

        return self.trajectories
