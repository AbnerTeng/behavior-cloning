from typing import Optional, Tuple

import gym
import numpy as np
import torch


class TradeEnv(gym.Env):
    """
    Customize trading environment, where the state is the OHLC data

    The environment has the following properties:
    - State shape: (20, 4) - 20 trading days of min-max normalized OHLC data
    - Action space: 4 - long, sell, short, and buy_to_cover
    - Reward is defined by the user based on the trading strategy

    - data shape: (2xx, 4) - 2xx trading days of OHLC data
    """
    def __init__(
        self,
        data: torch.Tensor,
        denorm_data: Optional[torch.Tensor],
        initial_balance: int = 1000000
    ) -> None:
        super().__init__()
        self.data = data
        self.denorm_data = denorm_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_idx = 20
        self.shares_held = 0
        self.trans_cost = 0.000  # 0.001
        self.have_position = False
        self.prev_act = None
        self.state_space = self.define_state_space()
        self.action_space = gym.spaces.Discrete(4)

    def define_state_space(self) -> gym.spaces.Space:
        """
        Define domain and shape of RL states

        Returns:
            gym.spaces.Space: state space
        """
        state_space = {}
        name = ['open', 'high', 'low', 'close']

        for i, n in enumerate(name):
            state_space[n] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.data[i].shape[0], ),
                dtype=np.float32
            )

        return gym.spaces.Dict(state_space)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Args:
            action (int): arg-maxed action from the model
        """
        close_today = self.denorm_data[self.current_idx, -1]  # 3 consists of the close price
        close_next = self.denorm_data[self.current_idx + 1, -1]

        share_today = np.floor(self.balance / close_today)
        cash_today = self.balance - share_today * close_today
        trans_cost_today = share_today * close_today * self.trans_cost
        balance_next = share_today * close_next + cash_today - trans_cost_today
        long_dr = (balance_next - self.balance) / self.balance
        short_dr = (-balance_next + self.balance) / self.balance
        dr = None

        if self.have_position is False:
            if action == 0:
                self.have_position = True
                dr = long_dr

            if action == 3:
                self.have_position = True
                dr = short_dr

            else:
                dr = 0

            self.prev_act = action

        if self.have_position is True:
            if self.prev_act == 0:

                if action in [0, 2]:
                    self.have_position = True
                    self.prev_act = 0
                    dr = long_dr

                if action in [1, 3]:
                    self.have_position = False
                    self.prev_act = action
                    dr = 0

            elif self.prev_act == 3:

                if action in [1, 3]:
                    self.have_position = True
                    self.prev_act = 3
                    dr = short_dr

                if action in [0, 2]:
                    self.have_position = False
                    self.prev_act = action
                    dr = 0

        self.balance = balance_next
        done = self.current_idx == self.data.shape[0] - 1
        obs = self.get_observation()

        return obs, dr, done, {}

    def reset(self) -> torch.Tensor:
        self.current_idx = 0
        self.balance = self.initial_balance
        self.shares_held = 0

        return self.get_observation()

    def get_observation(self) -> torch.Tensor:
        return self.data[self.current_idx: self.current_idx + 20]

    def calculate_reward(self) -> float:
        new_portfolio_value = self.balance + (
            self.shares_held * self.denorm_data[self.current_idx + 1, -1, -1]
        )
        reward = new_portfolio_value - (
            self.balance + (self.shares_held * self.denorm_data[self.current_idx, -1, -1])
        )

        return reward

    def render(self):
        pass
