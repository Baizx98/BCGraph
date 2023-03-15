import os.path as osp
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.loader import NeighborSampler


import quiver

dataset_path = osp.join(osp.dirname(
    osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(dataset_path)
data: Data = dataset[0]
train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)
csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = quiver.pyg.GraphSageSampler(
    csr_topo, sizes=[25, 10], device=0, mode='GPU')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    i = 0
    for seeds in train_loader:
        i = i+1
        print("time:", i)
        print("seeds:", seeds)
        print("seeds_length", seeds.size())
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        print("n_id:", len(n_id))
        print("batch_size:", batch_size)
        print("adjs:", adjs)
        print("adjs_count", len(adjs))


if __name__ == '__main__':
    # train()
    world_size = torch.cuda.device_count()
    a = [1]
    print(world_size)
    print(a)
