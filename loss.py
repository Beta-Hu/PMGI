import torch


def ints_loss(raw, fusion):
    loss = torch.mean((raw - fusion) ** 2)
    return loss


def grad_loss(raw, fusion):
    laplacian = torch.tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]]).to('cuda:0').float()
    grad_raw = torch.nn.functional.conv2d(raw, laplacian, stride=1, padding=1)
    grad_fusion = torch.nn.functional.conv2d(fusion, laplacian, stride=1, padding=1)
    loss = torch.mean((grad_raw - grad_fusion) ** 2)
    return loss
