import torch


def expectile_loss(diff: torch.Tensor, expectile: float = 0.99) -> torch.Tensor:
    """
    diff (torch.Tensor): difference between the target and the imp_loss_return
    expectile (float): expectile value (default: 0.99)
    """
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)
