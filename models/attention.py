import torch
import torch.nn.functional as F
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
    r : torch.Tensor
        Shape (batch_size, m, v_dim)
    """
    r = torch.mean(v, dim=1, keepdim=True) # shape [B, 1, v_dim]
    # Infer the number of target points from query
    m = q.shape[1]
    # Reshape the output
    r = torch.repeat_interleave(r, m, dim=1)
    return r


def laplace_attention(q, k, v, scale, normalize):
    """
    Laplace exponential attention

    Parameters
    ----------
    q : torch.Tensor
        Shape (batch_size, m, k_dim)

    k : torch.Tensor
        Shape (batch_size, n, k_dim)

    v : torch.Tensor
        Shape (batch_size, n, v_dim)

    scale : float
        scale in the L1 distance

    normalize : bool
        does the weights sum to 1?

    Returns
    -------
    r : torch.Tensor
        Shape (batch_size, m, v_dim)
    """
    k = k.unsqueeze(1) # shape [B, 1, n, k_dim]
    q = q.unsqueeze(2) # shape [B, m, 1, k_dim]
    unnorm_weights = - torch.abs((k - q) / scale) # shape [B, m, n, k_dim]
    unnorm_weights = torch.mean(weights, dim=-1) # shape [B, m, n]
    if normalize:
        weight_fn = F.softmax
    else:
        weight_fn = lambda x: 1 + torch.tanh(x)
    weights = weight_fn(unnorm_weights) # shape [B, m, n]
    r = torch.einsum('bij,bjk->bik', weights, v) # shape [B, m, v_dim]
    return r

def dot_product_attention(q, k, v, normalize):
    """
    Dot product attention

    Parameters
    ----------
    q : torch.Tensor
        Shape (batch_size, m, k_dim)

    k : torch.Tensor
        Shape (batch_size, n, k_dim)

    v : torch.Tensor
        Shape (batch_size, n, v_dim)

    normalize : bool
        does the weights sum to 1?

    Returns
    -------
    r : torch.Tensor
        Shape (batch_size, m, v_dim)
    """
    k_dim = q.shape[-1]
    scale = torch.sqrt(k_dim)
    unnorm_weights = tf.einsum('bik,bjk->bji', k, q) # shape [B, m, n]
    if normalize:
        weight_fn = F.softmax
    else:
        weight_fn = F.sigmoid
    weights = weight_fn(unnorm_weights) # shape [B, m, n]
    r = torch.einsum('bij,bjk->bik', weights, v) # shape [B, m, v_dim]


def multihead_attention(q, k, v, num_heads=8):
    """
    Dot product attention

    Parameters
    ----------
    q : torch.Tensor
        Shape (batch_size, m, k_dim)

    k : torch.Tensor
        Shape (batch_size, n, k_dim)

    v : torch.Tensor
        Shape (batch_size, n, v_dim)

    num_heads: int
        number of heads. Should divide v_dim.

    Returns
    -------
    r : torch.Tensor
        Shape (batch_size, m, v_dim)
    """
    raise NotImplementedError()


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

    def forward(self, q, k, v):
        if self._type == 'uniform':
            r = uniform_attention(q, v)
        elif self._type == 'laplace':
            r = laplace_attention(q, k, v, self._scale, self._normalise)
        elif self._type == 'dot_product':
            r = dot_product_attention(q, k, v, self._normalise)
        elif self._type == 'multihead':
            r = multihead_attention(q, k, v, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform', 'laplace', "
                             "'dot_product', 'multihead']"))

        return r
