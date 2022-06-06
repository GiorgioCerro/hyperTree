import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch import Tensor

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_dense_adj, add_self_loops
from geoopt.manifolds import PoincareBall
manifold_temp = PoincareBall()


class HyperbolicGraphConvolution(nn.Module):
    '''Hyperbolic graph convolution layer.
    '''
    def __init__(self, manifold, in_channels, out_channels,
        dropout=0.0, alpha=0.8, self_loops=True, use_activation=False,
        use_aggregation=False):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HyperbolicLinear(manifold, in_channels, out_channels)
        self.use_activation = use_activation
        self.use_aggregation = use_aggregation

        self.agg = None
        self.activation = None
        if self.use_aggregation == True:
            #self.agg = HyperbolicAggregation(manifold, out_channels)
            self.agg = HyperbolicLocalAgg(manifold, out_channels)
        if self.use_activation == True:
            self.activation = HyperbolicActivation(manifold, nn.LeakyReLU())
        #self.attention = HyperbolicAttention(manifold, out_channels, 
        #    dropout, alpha)
        
        self.self_loops = self_loops

    def forward(self, x, edge_index):
        if self.self_loops:
            edge_index = add_self_loops(edge_index)[0]
        adjacency = to_dense_adj(edge_index)[0].double() 
        adj_mask = adjacency>0

        h = self.linear.forward(x)
        if self.agg is not None:
            h = self.agg.forward(h, adj_mask)
            #h = self.attention.forward(h, adjacency)
        if self.activation is not None:
            h = self.activation.forward(h)
        return h

class HyperbolicLinear(nn.Module):
    '''Hyperbolic neural networks layer.
    '''
    def __init__(self, manifold, in_channels, out_channels, 
            dropout=0., bias=False):
        super(HyperbolicLinear, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.weights = Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

        '''
        self.weights = Parameter(manifold.proj(self.weights,0.5))
        _u, _s, _v = torch.svd(self.weights)
        _check = _s > (1 - 1e-4)
        _s[_check] = torch.rand(len(_s[_check]))
        _w = torch.mm(torch.mm(_u, torch.diag(_s)), _v.t())
        self.weights = Parameter(_w)
        '''
        self.weights = Parameter(manifold.proj(self.weights, 0.8))


    def reset_parameters(self):
        #init.normal_(self.weights, mean=0., std=0.2)
        init.xavier_uniform_(self.weights, gain = 1.)
        #zeros(self.bias)

    def forward(self,x):
        drop_weight = F.dropout(self.weights, self.dropout,
            training = self.training)
        #result = self.manifold.mobius_matvec(drop_weight, x)
        result = manifold_temp.mobius_matvec(drop_weight,x)
        #result = self.manifold.mobius_matvec(self.weights, x)
        '''
        if self.bias is not None:
            result = self.manifold.mobius_add(
                result, self.bias.repeat(len(result),1))
        '''
        result = self.manifold.proj(result)
        return result


class HyperbolicAttention(nn.Module):
    '''Hyperbolic aggregation layer
    '''

    def __init__(self, manifold, in_channels, dropout, alpha, concat=True):
        super(HyperbolicAttention, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.a = nn.Parameter(torch.empty(size=(2*in_channels, 1)))
        init.normal_(self.a, mean=0., std=0.2)
        self.a = nn.Parameter(manifold.proj(self.a, 0.9))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap(x) 
        e = self._prepare_attentional_mechanism_input(x_tangent)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        x_tangent_prime = torch.matmul(attention, x_tangent)
        
        #if self.concat:
        #    x_tangent_prime = F.elu(x_tangent_prime)

        x_hyperbolic = self.manifold.proj(
            self.manifold.expmap(x_tangent_prime))
        return x_hyperbolic

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.in_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.in_channels:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class HyperbolicActivation(nn.Module):
    '''Hyperbolic activation layer
    '''
    def __init__(self, manifold,  activation):
        super(HyperbolicActivation, self).__init__()
        self.manifold = manifold
        self.activation = activation

    def forward(self, x):
        x_tangent = self.manifold.logmap(x)
        x_activated = self.activation(x_tangent)
        x_hyperbolic = self.manifold.proj(self.manifold.expmap(x_activated))
        return x_hyperbolic


class HyperbolicAggregation(nn.Module):
    def __init__(self, manifold, in_channels):
        super(HyperbolicAggregation, self).__init__()
        self.manifold = manifold
        self.in_channels = in_channels

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap(x)
        #neigh = torch.sum(adj, axis=-1).view(-1,1)
        res = torch.matmul(adj, x_tangent) # / neigh
        res = self.manifold.proj(self.manifold.expmap(res))
        return res


class HyperbolicLocalAgg(nn.Module):
    def __init__(self, manifold, in_channels):
        super(HyperbolicLocalAgg, self).__init__()
        self.manifold = manifold
        self.in_channels = in_channels

    def forward(self, x, adj_mask):
        x_local = []
        for k in range(len(x)):
            x_local.append(self.manifold.expmap(torch.sum(
                self.manifold.logmap(
                    x[adj_mask[k]], x[k]), axis=0), x[k]).view(1,-1))
        x_local_agg = torch.cat(x_local)
        return x_local_agg
