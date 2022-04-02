import numpy as np
import torch
from contextlib import ExitStack
import glob
import graphicle as gcl

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from tqdm import tqdm
from heparchy.read.hdf import HdfReader

class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self,path,process_name):
        self.__process_name = process_name
        self.__files = glob.glob(path+'/*.hdf5')
        self.__ranges = [(-1,-1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            process = file_obj.read_process(name=self.__process_name)
            ini = self.__ranges[-1][1] + 1
            fin = ini + len(process) - 1
            self.__ranges.append((ini,fin))
        stack.close()

        __dtype = [('ini','i4'),('fin','i4')]
        self.__ranges = np.array(self.__ranges[1:],dtype=__dtype)


    def __len__(self):
        return self.__ranges['fin'][-1]


    #def __ranges__(self):
    #    return self.__ranges


    def __adj_matrix(self,eta,phi):
        _eta = eta - eta[...,None]
        _phi = phi - phi[...,None]
        _phi = np.min((_phi % (2*np.pi), np.abs(- _phi % (2*np.pi))), axis=0)
        return np.sqrt(_eta**2 + _phi**2)


    def __getitem__(self,idx):
        _file_idx = int(np.where(
            np.logical_and(np.less_equal(self.__ranges['ini'],idx),
                            np.greater_equal(self.__ranges['fin'],idx)))[0])

        _event_idx = idx - self.__ranges[_file_idx]['ini']
        with HdfReader(path=self.__files[_file_idx]) as hep_file:
            process = hep_file.read_process(name=self.__process_name)
            _event = process.read_event(_event_idx)

            graph = gcl.Graphicle.from_numpy(
                edges = _event.edges,
                pmu = _event.pmu,
                pdg = _event.pdg,
                final = _event.mask('final'))

            finals_pmu = graph.pmu[graph.final].data
            finals_hyper = _event.get_custom('hyper_coords')[graph.final.data]

            X = torch.tensor(
                np.transpose([finals_pmu['x'],finals_pmu['y'],finals_pmu['z'],
                    finals_pmu['e']]),dtype=torch.float64)

            adj_matrix = self.__adj_matrix(
                graph.pmu.eta[graph.final.data],
                    graph.pmu.phi[graph.final.data])
            knn = gcl.matrix.knn_adj(adj_matrix, k=10)
            knn = np.maximum(knn, np.transpose(knn))
            edges = np.array(np.where(knn == True))
            edge_index = torch.tensor(edges,dtype=torch.long)

            y = torch.tensor(finals_hyper,dtype=torch.float64)

            data = Data(x=X, edge_index=edge_index, y=y)
            return data
