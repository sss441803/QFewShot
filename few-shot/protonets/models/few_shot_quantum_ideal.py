from cirq import num_qubits
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torch
from torch import nn

from qtensor_ai.Quantum_Neural_Net import OptimalMetric, MetricLearning
from qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer

from protonets.models import register_model

from .utils import euclidean_dist


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.z_dim = encoder[-2][0].out_channels
        #self.z_dim = 5
        self.optimalmetric = OptimalMetric(self.z_dim, 2, optimizer=TamakiOptimizer(wait_time=20))

    # Added utility to evaluate quantum distance
    def quantum_dist(self, x, y):
        # x: N x D, N=n_class*n_query, D=z_dim
        # y: M X D, M=n_class*n_support, D=z_dim
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d).reshape(n*m, d)/6
        y = y.unsqueeze(0).expand(n, m, d).reshape(n*m, d)/6

        dist = self.optimalmetric(x, y)

        return torch.real(dist.reshape(n, m))

    # Added utility to evaluate quantum distance
    def cheat_quantum_dist(self, x, y):
        # x: N x D, N=n_class*n_query, D=z_dim
        # y: M X D, M=n_class*n_support, D=z_dim
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d).reshape(n*m, d)/2
        y = y.unsqueeze(0).expand(n, m, d).reshape(n*m, d)/2

        angle_difference = x - y
        cos_dist = torch.cos(angle_difference*torch.pi/2)
        dist = torch.prod(cos_dist, 1)

        return dist.reshape(n, m)  # util.euclidean returns this array shape as well
    
    def classical_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d).reshape(n*m, d)
        y = y.unsqueeze(0).expand(n, m, d).reshape(n*m, d)

        dist = self.cos(x, y)

        return dist.reshape(n, m)

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        #z = torch.remainder(z, 4)
        z_dim = z.size(-1)

        not_averaging = True
        if not_averaging:
            zs = z[:n_class*n_support] # zs: n_class*n_support X z_dim
            zq = z[n_class*n_support:] # zq: n_class*n_query X z_dim
            dists = self.quantum_dist(zq, zs) # dists: N X M, N=n_class*n_query, M=n_class*n_support
            dists = dists.reshape(n_class*n_query, n_class, n_support).mean(2) # dists: n_class*n_query X n_class
        else:
            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
            zq = z[n_class*n_support:]
            dists = self.quantum_dist(zq, z_proto)

        log_p_y = F.log_softmax(dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def MLP(z_dim, q_in):
        return nn.Sequential(
        nn.Linear(z_dim, z_dim//2),
        nn.ReLU(),
        nn.Linear(z_dim//2, z_dim//4),
        nn.ReLU(),
        nn.Linear(z_dim//4, z_dim//8),
        nn.ReLU(),
        nn.Linear(z_dim//8, q_in)
        )

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def conv_block_no_relu(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()#,
        #MLP(z_dim, 5)
    )

    return Protonet(encoder)
