"""
Search for the best action
"""
from typing import Optional, Tuple
import math

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from ..model.edt_model import ElasticDecisionTransformer


def sample_from_logits(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None
) -> torch.Tensor:
    """
    sampling logits with temperature and top_percentile
    """
    if top_percentile is not None:
        percentile = torch.quantile(
            logits, top_percentile, axis=-1, keepdim=True
        )
        logits = torch.where(logits >= percentile, logits, -np.inf)

    m = Categorical(logits=temperature * logits)

    return m.sample().unsqueeze(-1)


def expert_sampling(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None,
    expert_weight: Optional[float] = 10
) -> torch.Tensor:
    """
    Sample from expert policy
    """
    batch_size, seq_length, num_bin = logits.shape
    expert_logits = (
        torch.linspace(0, 1, num_bin).repeat(batch_size, seq_length, 1).to(logits.device)
    )
    return sample_from_logits(
        logits + expert_weight * expert_logits, temperature, top_percentile
    )


def mgdt_logits(
    logits: torch.Tensor, opt_weight: Optional[int] = 10
) -> torch.Tensor:
    logits_opt = torch.linspace(0.0, 1.0, logits.shape[-1]).to(logits.device)
    logits_opt = logits_opt.repeat(logits.shape[1], 1).unsqueeze(0)

    return logits + opt_weight * logits_opt


def return_search(
    model: ElasticDecisionTransformer,
    timesteps: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards_to_go: torch.Tensor,
    context_len: int,
    t: int,
    top_percentile: float,
    expert_weight: float,
    mgdt_sampling: bool = False,
    rs_steps: int = 2,
    rs_ratio: int = 1,
    real_rtg: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Search for the T for maximizing the return to go

    Args:
        model: (ElasticDecisionTransformer) --> model to use
        timesteps: (torch.Tensor) --> timesteps batch of test data (2D)
        states: (torch.Tensor) --> states batch of test data (2D)
        actions: (torch.Tensor) --> actions batch of test data (2D)
        rewards_to_go: (torch.Tensor) --> rewards to go batch of test data (2D)
        context_len: (int) --> context length
        t: (int) --> current timestep
        top_percentile: (float) --> top percentile for expert sampling
        expert_weight: (float) --> expert weight for expert sampling
        mgdt_sampling: (bool) --> whether to use mgdt sampling
        rs_steps: (int) --> number of steps for return search
        rs_ratio: (int) --> ratio for return search
        real_rtg: (bool) --> whether to use real return to go

    Returns:
        best_act: (torch.Tensor) --> best action to take
        context_len - best_i: (int) --> the index of the best action
    """
    # B x T x 1?
    highest_ret, best_i, best_act = -999, 0, None

    if t < context_len:
        for i in range(0, math.ceil((t + 1)/rs_ratio), rs_steps):
            act_preds, ret_preds, imp_ret_preds = model.get_action(
                states[:, i : context_len + i],
                actions[:, i : context_len + i],
                rewards_to_go[:, i : context_len + i],
                timesteps[:, i : context_len + i],
            )

            # first sample return with optimal weight
            # this sampling is the same as mgdt
            if mgdt_sampling:
                opt_rtg = expert_sampling(
                    mgdt_logits(ret_preds),
                    top_percentile=top_percentile,
                    expert_weight=expert_weight,
                )

                # we should estimate it again with the estimated rtg
                act_preds, ret_preds, imp_ret_preds_pure = model.get_action(
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    opt_rtg,
                    timesteps[:, i : context_len + i],
                )

            else:
                act_preds, ret_preds, imp_ret_preds_pure = model.get_action(
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    imp_ret_preds,
                    timesteps[:, i : context_len + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, t - i].detach().item()

            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                # best_act = act_preds[0, t - i].detach()
                best_act = act_preds.detach()


    else:
        for i in range(0, math.ceil(context_len/rs_ratio), rs_steps):
            act_preds, ret_preds, imp_ret_preds = model.get_action(
                states[:, t - context_len + 1 + i : t + 1 + i, :],
                actions[:, t - context_len + 1 + i : t + 1 + i, :],
                rewards_to_go[:, t - context_len + 1 + i : t + 1 + i],
                timesteps[:, t - context_len + 1 + i : t + 1 + i],
            )

            # first sample return with optimal weight
            if mgdt_sampling:
                opt_rtg = expert_sampling(
                    mgdt_logits(ret_preds),
                    top_percentile=top_percentile,
                    expert_weight=expert_weight,
                )

                # we should estimate the results again with the estimated return
                act_preds, ret_preds, imp_ret_preds_pure = model.get_action(
                    states[:, t - context_len + 1 + i : t + 1 + i, :],
                    actions[:, t - context_len + 1 + i : t + 1 + i, :],
                    opt_rtg,
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                )

            else:
                act_preds, ret_preds, imp_ret_preds_pure = model.get_action(
                    states[:, t - context_len + 1 + i : t + 1 + i, :],
                    actions[:, t - context_len + 1 + i : t + 1 + i, :],
                    imp_ret_preds,
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, -1 - i].detach().item()

            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                # best_act = act_preds[0, -1 - i].detach()
                best_act = act_preds.detach()

    return best_act, context_len - best_i


# def return_search_heuristic(
#     model,
#     timesteps,
#     states,
#     actions,
#     rewards_to_go,
#     rewards,
#     context_len,
#     t,
#     top_percentile,
#     expert_weight,
#     mgdt_sampling=False,
#     rs_steps=2,
#     rs_ratio=1,
#     real_rtg=False,
#     heuristic_delta=1,
#     previous_index=None,
# ):

#     # B x T x 1?
#     highest_ret = -999
#     best_i = 0
#     best_act = None

#     if t < context_len:
#         for i in range(0, math.ceil((t + 1)/rs_ratio), rs_steps):
#             _, act_preds, ret_preds, imp_ret_preds, _ = model.get_action(
#                 timesteps[:, i : context_len + i],
#                 states[:, i : context_len + i],
#                 actions[:, i : context_len + i],
#                 rewards_to_go[:, i : context_len + i],
#                 rewards[:, i : context_len + i],
#             )

#             if mgdt_sampling:
#                 opt_rtg = expert_sampling(
#                     mgdt_logits(ret_preds),
#                     top_percentile=top_percentile,
#                     expert_weight=expert_weight,
#                 )

#                 # we should estimate it again with the estimated rtg
#                 _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.get_action(
#                     timesteps[:, i : context_len + i],
#                     states[:, i : context_len + i],
#                     actions[:, i : context_len + i],
#                     opt_rtg,
#                     rewards[:, i : context_len + i],
#                 )

#             else:
#                 _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.get_action(
#                     timesteps[:, i : context_len + i],
#                     states[:, i : context_len + i],
#                     actions[:, i : context_len + i],
#                     imp_ret_preds,
#                     rewards[:, i : context_len + i],
#                 )

#             if not real_rtg:
#                 imp_ret_preds = imp_ret_preds_pure

#             ret_i = imp_ret_preds[:, t - i].detach().item()
#             if ret_i > highest_ret:
#                 highest_ret = ret_i
#                 best_i = i
#                 best_act = act_preds[0, t - i].detach()


#     else: # t >= context_len
#         prev_best_index = context_len - previous_index

#         for i in range(prev_best_index-heuristic_delta, prev_best_index+1+heuristic_delta):
#             if i < 0 or i >= context_len:
#                 continue
#             _, act_preds, ret_preds, imp_ret_preds, _ = model.get_action(
#                 timesteps[:, t - context_len + 1 + i : t + 1 + i],
#                 states[:, t - context_len + 1 + i : t + 1 + i],
#                 actions[:, t - context_len + 1 + i : t + 1 + i],
#                 rewards_to_go[:, t - context_len + 1 + i : t + 1 + i],
#                 rewards[:, t - context_len + 1 + i : t + 1 + i],
#             )

#             # first sample return with optimal weight
#             if mgdt_sampling:
#                 opt_rtg = expert_sampling(
#                     mgdt_logits(ret_preds),
#                     top_percentile=top_percentile,
#                     expert_weight=expert_weight,
#                 )

#                 # we should estimate the results again with the estimated return
#                 _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.get_action(
#                     timesteps[:, t - context_len + 1 + i : t + 1 + i],
#                     states[:, t - context_len + 1 + i : t + 1 + i],
#                     actions[:, t - context_len + 1 + i : t + 1 + i],
#                     opt_rtg,
#                     rewards[:, t - context_len + 1 + i : t + 1 + i],
#                 )

#             else:
#                 _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.get_action(
#                     timesteps[:, t - context_len + 1 + i : t + 1 + i],
#                     states[:, t - context_len + 1 + i : t + 1 + i],
#                     actions[:, t - context_len + 1 + i : t + 1 + i],
#                     imp_ret_preds,
#                     rewards[:, t - context_len + 1 + i : t + 1 + i],
#                 )

#             if not real_rtg:
#                 imp_ret_preds = imp_ret_preds_pure

#             ret_i = imp_ret_preds[:, -1 - i].detach().item()
#             if ret_i > highest_ret:
#                 highest_ret = ret_i
#                 best_i = i
#                 best_act = act_preds[0, -1 - i].detach()

#     return best_act, context_len - best_i
