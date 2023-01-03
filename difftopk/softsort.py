"""
Based on Prillo et al. (SoftSort: A Continuous Relaxation for the argsort Operator, ICML 2020)
"""
import torch


class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
