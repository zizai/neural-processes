import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .attention import Attention


class DeterministicEncoder(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(DeterministicEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)
        self.attention = Attention()

    def forward(self, x_context, y_context, x_target):
        """
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim)

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        Returns
        -------
        r : torch.Tensor
            Shape (batch_size, num_target, r_dim)
        """
        batch_size, num_context, _ = x_context.size()
        _, num_target, _ = x_target.size()

        # Flatten tensors, as encoder expects one dimensional inputs
        x_context_flat = x_context.view(batch_size * num_context, self.x_dim)
        y_context_flat = y_context.contiguous().view(batch_size * num_context, self.y_dim)

        input_pairs = torch.cat((x_context_flat, y_context_flat), dim=1)
        hidden_flat = self.input_to_hidden(input_pairs)

        # Reshape tensors into batches
        hidden = hidden_flat.view(batch_size, num_context, self.h_dim)

        # Get representation
        r = self.attention(x_target, x_context, hidden)
        return r


class LatentEncoder(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation s.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    s_dim : int
        Dimension of output representation s.
    """
    def __init__(self, x_dim, y_dim, h_dim, s_dim):
        super(LatentEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.s_dim = s_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim)]

        self.input_to_s = nn.Sequential(*layers)

    def forward(self, x_context, y_context):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_context, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        Returns
        -------
        s : torch.Tensor
            Shape (batch_size, s_dim)
        """
        batch_size, num_context, _ = x_context.size()

        # Flatten tensors, as encoder expects one dimensional inputs
        x_context_flat = x_context.view(batch_size * num_context, self.x_dim)
        y_context_flat = y_context.contiguous().view(batch_size * num_context, self.y_dim)

        # Encode each point into a representation s_i
        input_pairs = torch.cat((x_context_flat, y_context_flat), dim=1)
        s_i_flat = self.input_to_s(input_pairs)
        # Reshape tensors into batches
        s_i = s_i_flat.view(batch_size, num_context, self.s_dim)
        # Aggregate representations s_i into a single representation s
        s = torch.mean(s_i, dim=1)
        return s


class GaussianEncoder(nn.Module):
    """
    Perform reparameterization trick on s.
    We can sample from z, which is a distribution over s.

    Parameters
    ----------
    s_dim : int
        Dimension of state representation s.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, s_dim, z_dim):
        super(GaussianEncoder, self).__init__()

        self.s_dim = s_dim
        self.z_dim = z_dim

        self.s_to_hidden = nn.Linear(s_dim, s_dim)
        self.hidden_to_mu = nn.Linear(s_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(s_dim, z_dim)

    def forward(self, s):
        """
        Parameters
        ----------
        s : torch.Tensor
            Shape (batch_size, s_dim)

        Returns
        -------
        q_z : torch.distributions.Normal
        """
        hidden = torch.relu(self.s_to_hidden(s))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        q_z = Normal(mu, sigma)
        return q_z


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, r_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + r_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.rep_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, r, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        r : torch.Tensor
            Shape (batch_size, num_points, r_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)

        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        r_flat = r.view(batch_size * num_points, self.r_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)

        # Input is concatenation of r and z with every row of x
        input_pairs = torch.cat((x_flat, r_flat, z_flat), dim=1)
        hidden = self.rep_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        p_y = Normal(mu, sigma)
        return p_y


class ANP(nn.Module):
    """
    Attentive Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(ANP, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = DeterministicEncoder(x_dim, y_dim, h_dim, r_dim)
        self.xy_to_s = LatentEncoder(x_dim, y_dim, h_dim, z_dim)
        self.s_to_z = GaussianEncoder(h_dim, z_dim)
        self.xrz_to_y = Decoder(x_dim, r_dim, z_dim, h_dim, y_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if self.training:
            # Get deterministic code
            r = self.xy_to_r(x_context, y_context, x_target)

            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            s_context = self.xy_to_s(x_context, y_context)
            s_target = self.xy_to_s(x_target, y_target)
            z_prior = self.s_to_z(s_context)
            z_posterior = self.s_to_z(s_target)

            # Sample from encoded distribution using reparameterization trick
            z_sample = z_posterior.rsample()

            # Get output distribution
            p_y_pred = self.xrz_to_y(x_target, r, z_sample)

            return p_y_pred, z_posterior, z_prior
        else:
            # Get deterministic code
            r = self.xy_to_r(x_context, y_context, x_target)

            # At testing time, encode only context
            s_context = self.xy_to_s(x_context, y_context)
            z_prior = self.s_to_z(s_context)

            # Sample from the prior distribution
            z_sample = z_prior.rsample()

            # Predict target points based on context
            p_y_pred = self.xrz_to_y(x_target, r, z_sample)

            return p_y_pred
