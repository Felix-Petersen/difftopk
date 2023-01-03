import torch
from .difftopk import DiffTopkNet
from .neuralsort import NeuralSort
from .softsort import SoftSort
from typing import List


class TopKCrossEntropyLoss(torch.nn.Module):
    def __init__(
            self,
            diffsort_method: str,
            inverse_temperature: float,
            p_k: List[float],
            n: int,  # number of classes
            m: int = None,
            distribution: str = None,
            art_lambda: float = 0.25,
            device: str = 'cpu',
            top1_mode: str = 'sm',  # options [
            # 'sm',     (softmax for the top-1 component)
            # 'smce',   (extra softmax cross entropy loss for the top-1 component)
            # 'sort',   (using only the sorting result / permutation matrix, which is sometimes unstable)
            # ]
    ):
        super(TopKCrossEntropyLoss, self).__init__()
        self.diffsort_method = diffsort_method
        self.inverse_temperature = inverse_temperature
        self.p_k = p_k
        assert 0.9999 < sum(p_k) < 1.0001, 'P_K does not sum up to 1: {} sums up to {}.'.format(p_k, sum(p_k))
        self.k = len(p_k)
        self.device = device
        # self.use_softmax_for_top1 = use_softmax_for_top1
        self.top1_mode = top1_mode
        assert top1_mode in ['sm', 'smce', 'sort']
        self.n = n
        self.m = m
        if self.m is not None:
            assert self.k < self.m, (self.k, self.m)
            assert self.m < self.n, (self.m, self.n)
            num_inputs = self.m
        else:
            num_inputs = self.n

        if diffsort_method == 'bitonic':
            self.top_k_net = DiffTopkNet(
                'bitonic', num_inputs, self.k, sparse=True, device=device, steepness=inverse_temperature,
                art_lambda=art_lambda,
                distribution=distribution
            )

        elif diffsort_method == 'splitter_selection':
            self.top_k_net = DiffTopkNet(
                'splitter_selection', num_inputs, self.k, sparse=True, device=device, steepness=inverse_temperature,
                art_lambda=art_lambda,
                distribution=distribution
            )

        elif diffsort_method == 'bitonic__non_sparse':
            self.top_k_net = DiffTopkNet(
                'bitonic', num_inputs, self.k, sparse=False, device=device, steepness=inverse_temperature,
                art_lambda=art_lambda,
                distribution=distribution
            )

        elif diffsort_method == 'odd_even':
            self.top_k_net = DiffTopkNet(
                'odd_even', num_inputs, self.k, sparse=False, device=device, steepness=inverse_temperature,
                art_lambda=art_lambda,
                distribution=distribution
            )

        elif diffsort_method == 'neuralsort':
            self.sort_fn = NeuralSort(tau=1 / inverse_temperature)

        elif diffsort_method == 'softsort':
            the_sort_fn = SoftSort(tau=1 / inverse_temperature)
            self.sort_fn = lambda x: the_sort_fn(-x).transpose(-2, -1)

        # Sinkhorn sort is currently not supported as the used implementation was incompatible with current library
        # versions and required additional TensorFlow calls. For a current reference implementation (though in JAX)
        # we refer to the OTT library (https://github.com/ott-jax/ott).
        #
        # elif diffsort_method == 'sinkhorn_sort':  # in the original experiments 'sinkhorn_zto'
        #     sort_fn = SinkhornSort(epsilon=1 / inverse_temperature, sinkhorn_goal_mode='zero_to_one')

    def forward(self, outputs, labels):

        # Hard-select only the top-m elements to consider for the loss:
        if self.m is not None:
            assert len(outputs.shape) == 2, outputs.shape

            batch_size, num_classes = outputs.shape

            false_indices = torch.arange(num_classes).reshape(1, -1).to(self.device) != labels.unsqueeze(1)
            topnm1_indices = torch.topk(
                outputs[false_indices].reshape(batch_size, num_classes - 1),
                k=self.m - 1, dim=-1
            )[1]
            topnm1_outputs = outputs[false_indices].reshape(batch_size, num_classes - 1)[
                torch.arange(batch_size).to(self.device), topnm1_indices.T].T
            true_outputs = outputs[torch.arange(batch_size).to(self.device), labels].unsqueeze(1)

            outputs = torch.cat([true_outputs, topnm1_outputs], dim=1)
            labels = torch.zeros_like(labels)

        # --------------------------------------------------------------------------------------------------------------

        # Compute the differentiable permutation matrix P_topk
        if self.diffsort_method in ['neuralsort', 'sinkhorn_sort', 'softsort']:
            X = self.sort_fn(outputs)
            assert len(X.shape) == 3, X.shape
            assert X.shape[1] == outputs.shape[1], X.shape
            assert X.shape[2] == outputs.shape[1], X.shape
            P_topk = X[:, :, -self.k:]

        elif self.diffsort_method in ['bitonic', 'odd_even', 'bitonic__non_sparse', 'splitter_selection']:
            _, X = self.top_k_net(outputs)
            assert X.shape[1] == outputs.shape[1], X.shape
            assert X.shape[2] == self.k, X.shape
            P_topk = X[:, :, -self.k:]

        else:
            raise NotImplementedError(self.diffsort_method)

        # --------------------------------------------------------------------------------------------------------------

        if self.top1_mode == 'sort':
            topk_distribution = 0
            for k_, p_of_k_ in enumerate(self.p_k):
                k_ = k_ + 1
                topk_distribution = topk_distribution + p_of_k_ * P_topk[:, :, -k_:].sum(-1)
            loss = torch.nn.functional.nll_loss(torch.log(topk_distribution * (1 - 2e-7) + 1e-7), labels)

        elif self.top1_mode == 'sm':
            topk_distribution = self.p_k[0] * torch.softmax(outputs, dim=-1)
            for k_, p_of_k_ in enumerate(self.p_k[1:]):
                k_ = k_ + 2
                topk_distribution = topk_distribution + p_of_k_ * P_topk[:, :, -k_:].sum(-1)
            loss = torch.nn.functional.nll_loss(torch.log(topk_distribution * (1 - 2e-7) + 1e-7), labels)

        elif self.top1_mode == 'smce':
            topk_distribution = 0
            for k_, p_of_k_ in enumerate(self.p_k[1:]):
                k_ = k_ + 2
                topk_distribution = topk_distribution + p_of_k_ * P_topk[:, :, -k_:].sum(-1)
            loss_topk = torch.nn.functional.nll_loss(torch.log(topk_distribution * (1 - 2e-7) + 1e-7), labels)
            loss_top1 = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss = self.p_k[0] * loss_top1 + sum(self.p_k[1:]) * loss_topk

        else:
            raise NotImplementedError(self.top1_mode)

        return loss


