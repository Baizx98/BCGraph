# 测试不同图大小，不同采样器、参数得到的采样子图节点数
import os.path as osp

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler, NeighborLoader
import dgl
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import quiver

dataset_path_nvlink = "/home8t/bzx/data"
dataset_path_pagraph = "/home8t/bzx/padata"
dataset_name = "ogbn-products"
dataset_name = dataset_name.replace("-", "_")
print(dataset_name)
# pyg_loader = NeighborLoader()
edge_index = torch.load("/home8t/bzx/data/livejournal/edge_index.pt")
csr_topo: quiver.CSRTopo = quiver.CSRTopo(edge_index=edge_index)
print(csr_topo.edge_count)
print(csr_topo.node_count)
