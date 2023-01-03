import torch
from typing import List, Tuple
import math

try:
    import torch_sparse
    SPARSE_INSTALLED = True
except (ModuleNotFoundError, ImportError) as e:
    print(e)
    print('`torch-sparse` is not installed, which is ok, as long as no sparse modules are used.')
    SPARSE_INSTALLED = False

SORTING_NETWORK_TYPE = List[torch.tensor]


def s_best(x):
    return torch.clamp(x, -0.25, 0.25) + .5 + \
        ((x > 0.25).float() - (x < -0.25).float()) * (0.25 - 1/16/(x.abs()+1e-10))


class NormalCDF(torch.autograd.Function):
    def forward(ctx, x, sigma):
        ctx.save_for_backward(x, torch.tensor(sigma))
        return 0.5 + 0.5 * torch.erf(x / sigma / math.sqrt(2))

    def backward(ctx, grad_y):
        x, sigma = ctx.saved_tensors
        return grad_y * 1 / sigma / math.sqrt(math.pi * 2) * torch.exp(-0.5 * (x/sigma).pow(2)), None


def execute_topk(
        sorting_network,
        vectors,
        k,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'
):
    x = vectors
    alphas = []

    for split_a, split_b, combine_min, combine_max in sorting_network:
        a, b = x @ split_a.T, x @ split_b.T

        # float conversion necessary as PyTorch doesn't support Half for sigmoid as of 25. August 2021
        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == 'logistic':
            alpha = torch.sigmoid((b - a).type(new_type) * steepness).type(x.dtype)

        elif distribution == 'logistic_phi':
            alpha = torch.sigmoid(
                (b - a).type(new_type) * steepness / ((a - b).type(new_type).abs() + 1.e-10).pow(art_lambda)).type(
                x.dtype)

        elif distribution == 'gaussian':
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == 'reciprocal':
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == 'cauchy':
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + .5
            alpha = alpha.type(x.dtype)

        elif distribution == 'optimal':
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        w_min = alpha.unsqueeze(-2) * split_a.T + (1-alpha).unsqueeze(-2) * split_b.T
        w_max = (1-alpha).unsqueeze(-2) * split_a.T + alpha.unsqueeze(-2) * split_b.T
        x = (alpha * a + (1-alpha) * b) @ combine_min.T + ((1-alpha) * a + alpha * b) @ combine_max.T
        alphas.append((w_min, w_max))

    X = torch.eye(vectors.shape[1]).to(sorting_network[0][0].device).repeat(vectors.shape[0], 1, 1)[:, :, -k:]
    for current_alpha, (split_a, split_b, combine_min, combine_max) in zip(alphas[::-1], sorting_network[::-1]):
        X = (
              (current_alpha[1] @ (combine_max.T.unsqueeze(-3) @ X))
            + (current_alpha[0] @ (combine_min.T.unsqueeze(-3) @ X))
        )
    return None, X


def execute_sparse_topk(
        sorting_network,
        vectors,
        k,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'
):
    x = vectors
    alphas = []

    for split_a, split_b, combine_min, combine_max, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask in sorting_network:
        a, b = torch.mm(split_a.transpose(0, 1), x.T).transpose(0, 1), torch.mm(split_b.transpose(0, 1), x.T).transpose(0, 1)

        # float conversion necessary as PyTorch doesn't support Half for sigmoid as of 25. August 2021
        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == 'logistic':
            alpha = torch.sigmoid((b - a).type(new_type) * steepness).type(x.dtype)

        elif distribution == 'logistic_phi':
            alpha = torch.sigmoid(
                (b - a).type(new_type) * steepness / ((a - b).type(new_type).abs() + 1.e-10).pow(art_lambda)).type(
                x.dtype)

        elif distribution == 'gaussian':
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == 'reciprocal':
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == 'cauchy':
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + .5
            alpha = alpha.type(x.dtype)

        elif distribution == 'optimal':
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        x = torch.mm(combine_min.transpose(0, 1), (alpha * a + (1-alpha) * b).T).transpose(0, 1) + \
            torch.mm(combine_max.transpose(0, 1), ((1-alpha) * a + alpha * b).T).transpose(0, 1)
        alphas.append(alpha)

    X = torch.eye(vectors.shape[1]).to(sorting_network[0][0].device)[:, -k:].repeat(vectors.shape[0], 1, 1)
    X = X.reshape(-1, k)

    def get_sparse_permutation_matrix(alpha, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask):
        alpha = alpha[:, alpha_mask.long().squeeze(1)]
        alpha_duplicated = alpha.unsqueeze(2).repeat(1, 1, 2).reshape(alpha.shape[0], alpha.shape[1] * 2)
        ones = torch.ones(alpha.shape[0], remain_idxs.shape[0]).to(alpha.device)
        v = torch.cat([alpha_duplicated, 1-alpha_duplicated, ones], dim=1)
        i = torch.cat([alpha_idxs.T, one_minus_alpha_idxs.T, remain_idxs.T], dim=1)
        i = i.unsqueeze(0).repeat(alpha.shape[0], 1, 1).transpose(1, 2)
        i = i.reshape(i.shape[0] * i.shape[1], i.shape[2])
        i[:, 1] = i[:, 1] + torch.arange(0, vectors.shape[0] * vectors.shape[1], vectors.shape[1]).unsqueeze(0)\
                                .repeat(i.shape[0] // alpha.shape[0], 1).transpose(0, 1).reshape(-1).to(vectors.device)
        i[:, 0] = i[:, 0] + torch.arange(0, vectors.shape[0] * vectors.shape[1], vectors.shape[1]).unsqueeze(0)\
                                .repeat(i.shape[0] // alpha.shape[0], 1).transpose(0, 1).reshape(-1).to(vectors.device)
        v = v.reshape(-1)
        return i.T.long(), v, vectors.shape[0] * vectors.shape[1], vectors.shape[0] * vectors.shape[1]

    for current_alpha, (split_a, split_b, combine_min, combine_max, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask) in zip(alphas[::-1], sorting_network[::-1]):
        P = get_sparse_permutation_matrix(current_alpha, alpha_idxs, one_minus_alpha_idxs, remain_idxs, alpha_mask)
        X = torch_sparse.spmm(*P, X)

    X = X.reshape(vectors.shape[0], vectors.shape[1], k)
    return None, X


def topk(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors: torch.Tensor,
        k: int,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy',
        sparse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sorting_network[0][0].device == vectors.device, (
        f"The sorting network is on device {sorting_network[0][0].device} while the vectors are on device"
        f" {vectors.device}, but they both need to be on the same device."
    )
    if sparse:
        assert SPARSE_INSTALLED, '`torch-sparse` is necessary because the topk network is set to be sparse.'
        return execute_sparse_topk(
            sorting_network=sorting_network,
            vectors=vectors,
            k=k,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )
    else:
        return execute_topk(
            sorting_network=sorting_network,
            vectors=vectors,
            k=k,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )

