"""
Calculate the RankMe score (the higher, the better).

"RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank"
https://arxiv.org/abs/2210.02885

Notes:

* a random matrix with iid samples from a normal distribution is full rank (in the classical sense) but may not be
perfectly whitenened when using a finite number of samples, which RankMe measures.

* to remove potential issues, use a lot more samples than dimensions
"""

import torch
from torch import Tensor

DEFAULT_MAX_SAMPLES = 25_600  # number used in the paper
EPSILON = 1e-7  # suitable for float32


def calc_rankme(embeddings: Tensor, epsilon: float = EPSILON) -> float:
    """
    Calculate the RankMe score (the higher, the better).

    RankMe(Z) = exp (
        - sum_{k=1}^{min(N, K)} p_k * log(p_k)
    ),

    where p_k = sigma_k (Z) / ||sigma_k (Z)||_1 + epsilon
    where sigma_k is the kth singular value of Z.
    where Z is the matrix of embeddings

    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

    https://arxiv.org/pdf/2210.02885.pdf

    Args:
        embeddings: the embeddings to calculate the RankMe score for
        epsilon: the epsilon value to use for the calculation. The paper recommends 1e-7 for float32.

    Returns:
        the RankMe score
    """
    # compute the singular values of the embeddings
    _u, s, _vh = torch.linalg.svd(
        embeddings, full_matrices=False
    )  # s.shape = (min(N, K),)

    # normalize the singular values to sum to 1 [[Eq. 2]]
    p = (s / torch.sum(s, axis=0)) + epsilon

    # RankMe score is the exponential of the entropy of the singular values [[Eq. 1]]
    # this is sometimes called the `perplexity` in information theory
    entropy = -torch.sum(p * torch.log(p))
    rankme = torch.exp(entropy).item()

    return rankme


if __name__ == "__main__":
    dims = 100

    embeddings = torch.randn(DEFAULT_MAX_SAMPLES, dims)
    embeddings = embeddings / torch.norm(
        embeddings, dim=1, keepdim=True
    )  # unit-norm embeddings
    print(f"RankMe: {calc_rankme(embeddings)}")
