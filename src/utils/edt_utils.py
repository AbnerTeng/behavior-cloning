import torch


def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)
