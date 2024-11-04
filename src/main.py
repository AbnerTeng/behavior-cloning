"""
main script
"""
from argparse import ArgumentParser, Namespace
import numpy as np
from rich import print as rp
import torch
from torch.utils.data import DataLoader

from .model.dt_model import DecisionTransformer
from .dt_trainer import Trainer
from .create_dataset import (
    DataPreprocess,
    TradeLogDataset,
    EDTTradeLogDataset
)
from .prepare_dataset import PrepareDataset
from .utils.utils import (
    load_config,
    get_start_year_idx,
    get_slicev2,
    get_test_slicev2
)


def get_args() -> Namespace:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--state_with_vol",
        "-sv",
        action="store_true"
    )
    parser.add_argument(
        "--model",
        "-md",
        type=str,
        default="dt"
    )
    parser.add_argument(
        "--mode", '-m', type=str, default="test"
    )
    parser.add_argument(
        "--k", type=int, default=10
    )
    return parser.parse_args()


if __name__ == '__main__':
    p_args = get_args()
    train_config = load_config('config/train_cfg.yml')
    DEVICE = f"cuda:{train_config['device']}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(train_config['seed_number'])
    torch.cuda.manual_seed_all(train_config['seed_number'])
    torch.cuda.manual_seed(train_config['seed_number'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prepare_instance = PrepareDataset("trade_log/top", "env_data/sp500.csv")
    trajectories = prepare_instance.run(p_args.state_with_vol, p_args.k)
    new_trajectories = {}

    if p_args.model == "edt":
        keys = ['state', 'next_state', 'action', 'return', 'timestep']
    else:
        keys = ['state', 'action', 'return', 'timestep']

    for k in keys:
        new_trajectories[k] = np.array(
            [trajectory[k] for trajectory in trajectories.values()]
        )

    year_list = list(range(train_config['start_year'], train_config['end_year']))
    year_start_idx = get_start_year_idx(year_list, new_trajectories['timesteps'])
    state_size = new_trajectories['state'].shape[-1]
    action_size = new_trajectories['action'].shape[-1]

    for test_year in range(train_config["start_year"] + 1, train_config["end_year"]):
        rp(f"Test year: {test_year}")
        train_len = year_start_idx[test_year]
        max_len = train_config["train"]["max_len"]

        preproc = DataPreprocess(max_len, state_size, action_size, train_config["gamma"])
        state, next_state, action, timestep, returntogo, mask = preproc.split_data(
            new_trajectories['state'],
            new_trajectories['next_state'],
            new_trajectories['action'],
            new_trajectories['return']
        )

        t_train, s_train, ns_train, a_train, r_train = get_slicev2(
            [timestep, state, next_state, action, returntogo],
            p_args.k,
            start=0,
            end=train_len
        )
        # timestep = np.expand_dims(timestep.cpu().numpy(), axis=-1)
        t_test, s_test, ns_test, a_test, r_test = get_test_slicev2(
            [timestep, state, next_state, action, returntogo],
            p_args.k,
            train_len,
            max_len,
            year_start_idx,
            test_year,
            year_list
        )

        if p_args.model == "edt":
            train_dataset = EDTTradeLogDataset(s_train, ns_train, a_train, r_train, t_train, mask)

        else:
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
