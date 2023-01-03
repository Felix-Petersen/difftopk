"""
Based on Grover et al. (Stochastic Optimization of Sorting Networks via Continuous Relaxations, ICLR 2019)
"""
import torch


class NeuralSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = - scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.FloatTensor(dim, 1).fill_(1).to(scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.FloatTensor).to(scores.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau).transpose(-2, -1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=scores.device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(dim0=1, dim1=0).flatten().long().to(scores.device)
            r_idx = torch.arange(dim).repeat([bsize, 1]).flatten().long().to(scores.device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat

