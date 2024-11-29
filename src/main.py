"""
main script
"""
import os

import numpy as np
import torch
from rich import print as rp
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from .create_dataset import (
    DataPreprocess,
    TradeLogDataset,
)
from .model.model import DecisionTransformer, ElasticDecisionTransformer
from .prepare_dataset import PrepareDataset
from .trainer import Trainer
from .utils.utils import get_slicev2, get_start_year_idx, get_test_slicev2, load_config, set_all_seed


if __name__ == '__main__':
    cfg = load_config('config/train_cfg.yml')
    DEVICE = f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
    set_all_seed(cfg.seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prepare_instance = PrepareDataset(f"trade_log/{cfg.expr}", "env_data/sp500.csv")
    trajectories = prepare_instance.run(cfg.state_with_vol, k=cfg.k)
    new_trajectories = {}
    keys = ["state", "next_state", "action", "return", "timestep"]

    for k in keys:
        new_trajectories[k] = np.array(
            [trajectory[k] for trajectory in trajectories.values()]
        )

    year_list = list(range(**cfg.year_range))
    year_start_idx = get_start_year_idx(year_list, new_trajectories['timestep'])
    state_size = new_trajectories['state'].shape[-1]
    action_size = new_trajectories['action'].shape[-1]

    for test_year in range(cfg.year_range[0] + 1, cfg.year_range[-1]):
        rp(f"Test year: {test_year}")
        train_len = year_start_idx[test_year]
        max_len = cfg.train.max_len

        preproc = DataPreprocess(max_len, state_size, action_size, DEVICE, cfg.gamma)

        state, norm_state, next_state, action, timestep, returntogo, mask = preproc.split_data(
            new_trajectories['state'],
            new_trajectories['next_state'],
            new_trajectories['action'],
            new_trajectories['return']
        )

        t_train, norm_s_train, ns_train, a_train, r_train = get_slicev2(
            [timestep, norm_state, next_state, action, returntogo],
            cfg.k,
            start=0,
            end=train_len
        )
        t_test, s_test, norm_s_test, ns_test, a_test, r_test = get_test_slicev2(
            [timestep, state, norm_state, next_state, action, returntogo],
            cfg.k,
            train_len,
            max_len,
            year_start_idx,
            test_year,
            year_list
        )
        train_dataset = TradeLogDataset(norm_s_train, ns_train, a_train, r_train, t_train, mask)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size * cfg.k,
            shuffle=True
        )

        if cfg.model == "edt":
            model = ElasticDecisionTransformer(
                state_size,
                action_size,
                cfg.train.hidden_size,
                3,
                1,
                True
            )
        else:
            model = DecisionTransformer(
                state_size,
                action_size,
                cfg.train.hidden_size
            )

        if test_year == cfg.year_range[0] + 1:
            if not os.path.exists(f'ckpts/{cfg.model}/{cfg.expr}_model_weights'):
                os.makedirs(f'ckpts/{cfg.model}/{cfg.expr}_model_weights')

            torch.save(model.state_dict(), f'ckpts/{cfg.model}/{cfg.expr}_model_weights/original.pth')

        trainer = Trainer(
            test_year,
            model,
            max_len,
            cfg.expr,
            cfg.model,
            is_elastic=True if cfg.model == "edt" else False
        )

        if cfg.mode == "train":
            trainer.train(
                cfg.train.epochs,
                train_dataloader,
                DEVICE,
                cfg.optimizer,
                expectile=0.99,
                state_loss_weight=1.0,
                exp_loss_weight=0.5,
                ce_weight=0.001
            )

        elif cfg.mode == "test":
            DEVICE="cpu"
            s_test = s_test.to(DEVICE)
            norm_s_test = norm_s_test.to(DEVICE)
            first_20 = norm_s_test[0]
            single_state = [
                norm_s_test[cfg.k * i, -1, :].unsqueeze(0) for i in range(1, int(norm_s_test.shape[0] / cfg.k))
            ]
            single_state = torch.stack(single_state).to(DEVICE)
            single_state = torch.cat(
                [
                    first_20,
                    single_state.squeeze(1)
                ],
                dim=0
            )
            first_20_denorm = s_test[0]
            single_state_denorm = [
                s_test[cfg.k * i, -1, :].unsqueeze(0) for i in range(1, int(s_test.shape[0] / cfg.k))
            ]
            single_state_denorm = torch.stack(single_state_denorm).to(DEVICE)
            single_state_denorm = torch.cat(
                [
                    first_20_denorm,
                    single_state_denorm.squeeze(1)
                ],
                dim=0
            )

            if cfg.model == "edt":
                trainer.edt_test(
                    DEVICE,
                    cfg.expr,
                    state_size,
                    action_size,
                    single_state,
                    single_state_denorm,
                    cfg.train.target_return,
                    heuristic=cfg.heuristic_search
                )
            else:
                trainer.dt_test(
                    DEVICE,
                    cfg.expr,
                    state_size,
                    action_size,
                    single_state,
                    cfg.train.target_return,
                )

        else:
            rp('Invalid mode')
