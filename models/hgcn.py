import torch 
from layers.hyp_layer import HyperbolicGraphConvolution as HypLayer

class HyperGCN(torch.nn.Module):
    def __init__(self, manifold, _in, _h1, _out):
        super().__init__()
        self.manifold = manifold
        self.hconv1 = HypLayer(manifold, _in-1, _h1, use_activation=False)
        self.hconv_2 = HypLayer(manifold, _h1, _h1, use_activation=False)
        '''
        self.hconv_3 = HypLayer(manifold, _h1, _h1, use_activation=False)
        self.hconv_4 = HypLayer(manifold, _h1, _h1, use_activation=True)
        self.hconv_5 = HypLayer(manifold, _h1, _h1, use_activation=True)
        '''
        self.hconv2 = HypLayer(manifold, _h1, _h1, use_activation=False,
            use_aggregation=True)
        self.hconv3 = HypLayer(manifold, _h1, _h1, use_activation=False,
            use_aggregation=True)
        self.hconv4 = HypLayer(manifold, _h1, _out, use_activation=False)

    def forward(self, data):
        x = self.manifold.lorentz_to_poincare(data.x)
        edge_index = data.edge_index

        x = self.hconv1(x, edge_index)
        x = self.hconv_2(x, edge_index)
        #x = self.hconv_3(x, edge_index)
        #x = self.hconv_4(x, edge_index)
        #x = self.hconv_5(x, edge_index)
        x = self.hconv2(x, edge_index)
        x = self.hconv3(x, edge_index)
        x = self.hconv4(x, edge_index)

        return x
