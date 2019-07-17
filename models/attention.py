import torch

from torch import nn


def uniform_attention(q, v):
    """
    Uniform attention. Equivalent to Neural Process.

    Parameters
    ----------
    q : torch.Tensor
        Shape (batch_size, m, k_dim)

    v : torch.Tensor
        Shape (batch_size, n, v_dim)

    Returns
    -------
    tensor of shape (batch_size, v_dim).
    """
    m = q.shape[1]
    out = torch.mean(v, dim=1) # shape [B, 1, v_dim]
    #out = torch.repeat_interleave(out, m, dim=1)
    return out


class Attention(nn.Module):
    """
    The Attention module

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.

    Parameters
    ----------
    rep : string
        transformation to apply to contexts before computing attention.
        One of: ['identity','mlp'].

    att_type : string
        type of attention. One of the following:
        ['uniform','laplace','dot_product','multihead']

    scale: float
        length scale in Laplace attention.

    normalise: bool
        determining whether to:
        1) apply softmax to weights so that they sum to 1 across context pts or
        2) apply custom transformation to have weights in [0,1].

    num_heads: int
        number of heads for multihead.
    """

    def __init__(self, att_type='uniform', scale=1., normalise=True, num_heads=8):
        super(Attention, self).__init__()
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == 'multihead':
            self._num_heads = num_heads

    def forward(self, k, q, r):
        rep = uniform_attention(q, r)

        """
        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = multihead_attention(q, k, r, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform', 'laplace', "
                             "'dot_product', 'multihead']"))
        """

        return rep
