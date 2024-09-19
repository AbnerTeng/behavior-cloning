from argparse import ArgumentParser
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
from .utils.utils import load_config



def parsing_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", '-m', type=str, default="train"
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

    # load and split data
    action_data = np.load('/home/minyu/Financial_RL/dt/new_expert/action.npy')
    action_data1 = np.load('/home/minyu/Financial_RL/dt/new_expert/action1.npy')
    feat_data = np.load('/home/minyu/Financial_RL/all_feat.npy')
    return_data = np.load('/home/minyu/Financial_RL/dt/new_expert/daily_return.npy', allow_pickle=True).astype(float)
    return_data1 = np.load('/home/minyu/Financial_RL/dt/new_expert/daily_return1.npy', allow_pickle=True).astype(float)
    date_data = np.load('/home/minyu/Financial_RL/all_date.npy', allow_pickle=True)

    return_data[np.isnan(return_data)] = 0
    return_data1[np.isnan(return_data1)] = 0

    for idx, date in enumerate(date_data):
        if str(train_config['start_year']) in date:
            cut_idx = idx
            date_data = date_data[cut_idx:]
            feat_data = feat_data[cut_idx:]
            return_data = return_data[cut_idx:]
            return_data1 = return_data1[cut_idx:]
            break

    year_list = list(range(train_config['start_year'], train_config['end_year']))
    year_start_idx = {}
    for y in year_list:
        for idx, date in enumerate(date_data):
            if str(y) in date:
                year_start_idx[y] = idx
                break

    # hyperparameters
    state_size = feat_data.shape[1]
    action_size = action_data.shape[1]

    for test_year in range(train_config['start_year'] + 1, train_config['end_year']):
        rp(f'Test year: {test_year}')
        train_len = year_start_idx[test_year]

        if test_year != 2022:
            date_train, date_test = date_data[:train_len], date_data[train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]]
            state_train, state_test, action_train, action_test, return_train, return_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]], action_data[:train_len], action_data[train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]], return_data[:train_len], return_data[train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]]
            state1_train, state1_test, action1_train, action1_test, return1_train, return1_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]], action_data1[:train_len], action_data1[train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]], return_data1[:train_len], return_data1[train_len-(train_config['train']['max_len']-1):year_start_idx[test_year+1]]

        else:
            date_train, date_test = date_data[:train_len], date_data[train_len-(train_config['train']['max_len']-1):]
            state_train, state_test, action_train, action_test, return_train, return_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(train_config['train']['max_len']-1):], action_data[:train_len], action_data[train_len-(train_config['train']['max_len']-1):], return_data[:train_len], return_data[train_len-(train_config['train']['max_len']-1):]
            state1_train, state1_test, action1_train, action1_test, return1_train, return1_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(train_config['train']['max_len']-1):], action_data1[:train_len], action_data1[train_len-(train_config['train']['max_len']-1):], return_data1[:train_len], return_data1[train_len-(train_config['train']['max_len']-1):]

        preproc = DataPreprocess(train_config['train']['max_len'], state_size, action_size, train_config['gamma'])
        s_train, a_train, t_train, rtg_train, mask_train = preproc.split_data(state_train, action_train, return_train)
        s1_train, a1_train, t1_train, rtg1_train, mask1_train = preproc.split_data(state1_train, action1_train, return1_train)
        s_train, a_train, t_train, rtg_train, mask_train = torch.cat((s_train, s1_train), axis=0), torch.cat((a_train, a1_train), axis=0), torch.cat((t_train, t1_train), axis=0), torch.cat((rtg_train, rtg1_train), axis=0), torch.cat((mask_train, mask1_train), axis=0)
        train_dataset = TradeLogDataset(s_train, a_train, t_train, rtg_train, mask_train)
        train_dataloader = DataLoader(train_dataset, batch_size=train_config['train']['batch_size'], shuffle=True)

        dt_model = DecisionTransformer(
            state_size,
            action_size,
            train_config['train']['hidden_size'],
            train_config['train']['max_len']
        ).to(DEVICE)

        if test_year == train_config['start_year']+1:
            torch.save(dt_model.state_dict(), './model_weights/original.pth')

        trainer = Trainer(test_year, dt_model, train_config['train']['max_len'])

        if p_args.mode == "train":
            trainer.train(train_config['train']['epochs'], train_dataloader)

        elif p_args.mode == "test":
            Trainer.test(DEVICE, state_size, action_size, state_test, train_config['train']['target_return'])

        else:
            rp('Invalid mode')
