import torch
from .functional import topk
from .networks import get_sorting_network


class DiffTopkNet(torch.nn.Module):
    def __init__(
        self,
        sorting_network_type: str,
        size: int,
        k: int,
        sparse: bool = False,
        device: str = 'cpu',
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = 'cauchy',
    ):
        super(DiffTopkNet, self).__init__()
        self.sorting_network_type = sorting_network_type
        self.size = size
        self.k = k
        self.sparse = sparse

        if sparse:
            self.sorting_network = get_sorting_network('sparse_' + sorting_network_type, size, device, k)
        else:
            self.sorting_network = get_sorting_network(sorting_network_type, size, device)

        if interpolation_type is not None:
            assert distribution is None or distribution == 'cauchy' or distribution == interpolation_type, (
                'Two different distributions have been set (distribution={} and interpolation_type={}); however, '
                'they have the same interpretation and interpolation_type is a deprecated argument'.format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type
            print('Warning: `interpolation_type` is a deprecated argument, please use `distribution` instead.')

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution
        self.sparse = sparse

    def forward(self, vectors):
        assert len(vectors.shape) == 2, vectors.shape
        assert vectors.shape[1] == self.size, (vectors.shape, self.size)
        return topk(
            self.sorting_network, vectors, self.k, self.steepness, self.art_lambda, self.distribution, sparse=self.sparse
        )
