"""
main script
"""
from argparse import ArgumentParser, Namespace
import numpy as np
from rich import print as rp
import torch
from torch.utils.data import DataLoader

from .model.rl_model import DecisionTransformer
from .trainer import Trainer
from .create_dataset import (
    DataPreprocess,
    TradeLogDataset
)
from .prepare_dataset import PrepareDataset
from .utils.utils import (
    load_config,
    get_slicev2,
    get_test_slicev2
)


def parsing_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", '-m', type=str, default="test"
    )
    parser.add_argument(
        "--k", type=int, default=10
    )
    return parser.parse_args()


if __name__ == '__main__':
    p_args = parsing_args()
    train_config = load_config('config/train_cfg.yml')
    DEVICE = f"cuda:{train_config['device']}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train_config['seed_number'])
    torch.cuda.manual_seed_all(train_config['seed_number'])
    torch.cuda.manual_seed(train_config['seed_number'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prepare_instance = PrepareDataset("trade_log/top", "sp500.csv")
    trajectories = prepare_instance.run(p_args.k)
    states = np.array([trajectory['state'] for trajectory in trajectories.values()])[:, :, :4]  # (k, d, 4)
    actions = np.array([trajectory['action'] for trajectory in trajectories.values()])  # (k, d, 4)
    returns = np.array([trajectory['return'] for trajectory in trajectories.values()])  # (k, d)
    timesteps = np.array([trajectory['timestep'] for trajectory in trajectories.values()])  # (k, d)
    year_list = list(range(train_config['start_year'], train_config['end_year']))
    year_start_idx = {}

    for y in year_list:
        for idx, date in enumerate(timesteps[-1]):
            if str(y) in date:
                year_start_idx[y] = idx
                break

    state_size, action_size = states.shape[-1], actions.shape[-1]

    for test_year in range(train_config["start_year"] + 1, train_config["end_year"]):
        rp(f"Test year: {test_year}")
        train_len = year_start_idx[test_year]
        max_len = train_config["train"]["max_len"]

        preproc = DataPreprocess(max_len, state_size, action_size, train_config["gamma"])
        state, action, timestep, returntogo, mask = preproc.split_data(states, actions, returns)
        # timestep = np.expand_dims(timestep.cpu().numpy(), axis=-1)
        t_train = get_slicev2(
            timestep, p_args.k, max_len=max_len, start=0, end=train_len
        )  # (tr_len * k, window_size, 1)
        s_train = get_slicev2(
            state, p_args.k, max_len=max_len, start=0, end=train_len
        )  # (tr_len * k, window_size, 4)
        a_train = get_slicev2(
            action, p_args.k, max_len=max_len, start=0, end=train_len
        )  # (tr_len * k, window_size, 4)
        r_train = get_slicev2(
            returntogo, p_args.k, max_len=max_len, start=0, end=train_len
        )  # (tr_len * k, window_size)

        t_test = get_test_slicev2(
            timestep, p_args.k, train_len, max_len, year_start_idx, test_year, year_list
        )  # (te_len * k, window_size, 1)
        s_test = get_test_slicev2(
            state, p_args.k, train_len, max_len, year_start_idx, test_year, year_list
        )
        a_test = get_test_slicev2(
            action, p_args.k, train_len, max_len, year_start_idx, test_year, year_list
        )
        r_test = get_test_slicev2(
            returntogo, p_args.k, train_len, max_len, year_start_idx, test_year, year_list
        )

        train_dataset = TradeLogDataset(s_train, a_train, t_train, r_train, mask)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config["train"]["batch_size"] * p_args.k,
            shuffle=True
        )

        dt_model = DecisionTransformer(
            state_size,
            action_size,
            train_config['train']['hidden_size']
        ).to(DEVICE)

        if test_year == train_config['start_year'] + 1:
            torch.save(dt_model.state_dict(), 'model_weights/original.pth')
        trainer = Trainer(test_year, dt_model, max_len)

        if p_args.mode == "train":
            trainer.train(train_config['train']['epochs'], train_dataloader)

        elif p_args.mode == "test":
            first_20 = s_test[0]
            single_state = [
                s_test[p_args.k * i, -1, :].unsqueeze(0) for i in range(1, int(s_test.shape[0] / p_args.k))
            ]
            single_state = torch.stack(single_state).to(DEVICE)
            single_state = torch.cat(
                [
                    first_20,
                    single_state.squeeze(1)
                ],
                dim=0
            )
            trainer.test(
                DEVICE,
                state_size,
                action_size,
                single_state,
                train_config['train']['target_return']
            )

        else:
            rp('Invalid mode')
