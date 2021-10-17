from typing import List, cast, NamedTuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from custom_types import IStateDict
from util import flatten_weight, split_state_dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ITorchReturnTypeMax = NamedTuple('torch_return_type_max', [(
    'indices', torch.Tensor), ('values', torch.Tensor)])


def _icc_loss(pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    kl_loss_helper = nn.KLDivLoss(reduction="batchmean")
    _sum = 0.0

    for helper_pred in helper_preds:
        _sum += kl_loss_helper(pred, helper_pred).float()

    return _sum / len(helper_preds)


def _transform_onehot(tensor: torch.Tensor) -> torch.Tensor:
    max_values = cast(torch.Tensor, torch.max(
        tensor, dim=1, keepdim=True).values)
    return (tensor >= max_values).float() - \
        torch.sigmoid(tensor - max_values).detach() + \
        torch.sigmoid(tensor - max_values)


def _calculate_pseudo_label(local_pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    _sum = torch.zeros_like(local_pred)
    for pred in [local_pred, *helper_preds]:
        one_hot = _transform_onehot(pred)
        _sum += one_hot

    return torch.argmax(_sum, dim=1)


def _consistency_regularization(
    pred: torch.Tensor,
    pred_noised: torch.Tensor,
    helper_preds: List[torch.Tensor]
):

    pseudo_label = _calculate_pseudo_label(
        pred_noised, helper_preds).type(torch.LongTensor).to(device)

    pseudo_label_CE_loss = F.cross_entropy(
        pred_noised, pseudo_label)
    kl_loss = _icc_loss(pred, helper_preds)

    return pseudo_label_CE_loss + kl_loss


def iccs_loss(
    pred: torch.Tensor,
    pred_noised: torch.Tensor,
    helper_preds: List[torch.Tensor],
    lambda_iccs: float
):
    return _consistency_regularization(pred, pred_noised, helper_preds) * lambda_iccs


def regularization_loss(sigma: Iterable[Parameter], psi: Iterable[Parameter], lambda_l1: float, lambda_l2: float):
    sigma = list(sigma)
    psi = list(psi)

    loss = 0.0
    for idx in range(len(sigma)):
        loss += torch.sum(((sigma[idx] - psi[idx]) ** 2) * lambda_l2)
        loss += torch.sum(torch.abs(psi[idx]) * lambda_l1)

    return loss
