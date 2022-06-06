import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn.functional as F
import torch

#from optimizer.radam import RiemannianAdam
from geoopt.optim.radam import RiemannianAdam
from manifold.poincare import PoincareBall
#from geoopt.manifolds import PoincareBall
manifold = PoincareBall()


def distance_matrix(nodes):
    length = len(nodes)
    matrix = torch.zeros(length,length)
    for n_idx in range(len(nodes)):
        matrix[n_idx] = manifold.distance(
            torch.unsqueeze(nodes[n_idx],0), nodes) + 1e-8
    matrix = matrix[torch.triu(torch.ones(length, length), diagonal=1) == 1]
    return matrix**2.


class LitHGCN(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.hyp_gcn = model(manifold, 4, 10, 2).double()
        self.lr = lr
        #self.hyp_gcn = model(manifold, 4, 2).double()

    def training_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        counts = torch.unique(batch.batch, return_counts=True)[1]
        sums = [0]
        for c in counts: sums.append(c + sums[-1])
        for j in range(len(sums) - 1):
            batch.y[sums[j] : sums[j+1]] += sums[j]

        loss_temp = self.loss_function(output, batch.y)
        
        self.log('training loss', loss_temp, 
            prog_bar=True, batch_size=batch.num_graphs)

        self.logger.experiment.add_scalar('loss/train', loss_temp,
            self.global_step)

        return loss_temp
    

    def validation_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        counts = torch.unique(batch.batch, return_counts=True)[1]
        sums = [0]
        for c in counts: sums.append(c + sums[-1])
        for j in range(len(sums) - 1):
            batch.y[sums[j] : sums[j+1]] += sums[j]

        loss_temp = self.loss_function(output, batch.y)
                
        self.log('validation loss', loss_temp, batch_size=batch.num_graphs)


    def test_step(self, batch, batch_idx):
        output = self.hyp_gcn(batch)

        counts = torch.unique(batch.batch, return_counts=True)[1]
        sums = [0]
        for c in counts: sums.append(c + sums[-1])
        for j in range(len(sums) - 1):
            batch.y[sums[j] : sums[j+1]] += sums[j]

        loss_temp = self.loss_function(output, batch.y)
                
        self.log('test loss', loss_temp, batch_size=batch.num_graphs)


    def configure_optimizers(self):
        optimizer = RiemannianAdam(self.hyp_gcn.parameters(),
            lr=self.lr, weight_decay=5e-4, stabilize=1)
        return optimizer


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss/epoch', avg_loss,
            self.current_epoch)


    def log_sigmoid(self, vector):
        return torch.log(( 1 / (1 + torch.exp(-vector))))
    

    def loss_function(self, output, neighs):
        pos_neigh, neg_neigh = neighs[:,0], neighs[:,1:]
        pos_dist = manifold.distance(output, output[neighs[:,0]])**2
        neg_dist1= manifold.distance(output, output[neighs[:,1]])**2
        neg_dist2 = manifold.distance(output, output[neighs[:,2]])**2
       
        pos_loss = self.log_sigmoid(- pos_dist)
        neg_loss = self.log_sigmoid(neg_dist1) + self.log_sigmoid(neg_dist2)

        loss = - torch.mean(pos_loss + neg_loss)
        return loss


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running training ...')
        return bar
