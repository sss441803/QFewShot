import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os
import json
import tqdm

import protonets.utils.data as data_utils
from protonets.utils import filter_opt

def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    sample = next(iter(data['test']))

    xs = Variable(sample['xs']) # support
    xq = Variable(sample['xq']) # query

    n_class = xs.size(0)
    assert xq.size(0) == n_class
    n_support = xs.size(1)
    n_query = xq.size(1)

    x = None
    for i in range(1000):
        try:
            sample = next(iter(data['test']))
            xs = Variable(sample['xs'])
            if x != None:
                x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), x], 0)
            else:
                x = xs.view(n_class * n_support, *xs.size()[2:])
        except TypeError:
            break

    print(x.shape)

    #target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    #target_inds = Variable(target_inds, requires_grad=False)

    #if xq.is_cuda:
    #    target_inds = target_inds.cuda()

    #x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
    #                xq.view(n_class * n_query, *xq.size()[2:])], 0)

    z = model.encoder.forward(x)
    np.save('z', z.detach().cpu())
    #z_dim = z.size(-1)
'''
    not_averaging = False
    if not_averaging:
        zs = z[:n_class*n_support] # zs: n_class*n_support X z_dim
        zq = z[n_class*n_support:] # zq: n_class*n_query X z_dim
        dists = model.quantum_dist(zq, zs) # dists: N X M, N=n_class*n_query, M=n_class*n_support
        dists = dists.reshape(n_class*n_query, n_class, n_support).mean(2).abs() # dists: n_class*n_query X n_class
    else:
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]
        dists = model.quantum_dist(zq, z_proto).abs()

    log_p_y = F.log_softmax(dists, dim=1).view(n_class, n_query, -1)
    _, y_hat = log_p_y.max(2)

    np.save('xq', xq.detach().cpu())
    np.save('xs', xs.detach().cpu())
    for query in range(zq.shape[0]):
        np.savetxt('zq{}'.format(query), zq[query].reshape(model.n_qubits, model.n_layers).detach().cpu())
    for way in range(z_proto.shape[0]):
        np.savetxt('z_proto{}.txt'.format(way), z_proto[way].reshape(model.n_qubits, model.n_layers).detach().cpu())
    np.savetxt('y_hat.txt', y_hat.detach().cpu())
'''