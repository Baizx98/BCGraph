import os.path as osp
import os
from typing import Tuple
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.loader import NeighborSampler

import pandas as pd


import quiver

dataset_path = osp.join(osp.dirname(
    osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(dataset_path)
data: Data = dataset[0]
train_idx: torch.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)
train_idx_parallel = train_idx.split(train_idx.size(0)//2)
print(train_idx_parallel[0][:10])
print(train_idx_parallel[1][:10])

train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)
train_loader_gpu0 = torch.utils.data.DataLoader(train_idx_parallel[0],
                                                batch_size=1024,
                                                shuffle=True,
                                                drop_last=True)
train_loader_gpu1 = torch.utils.data.DataLoader(train_idx_parallel[1],
                                                batch_size=1024,
                                                shuffle=True,
                                                drop_last=True)

csr_topo = quiver.CSRTopo(data.edge_index)
quiver_sampler = quiver.pyg.GraphSageSampler(
    csr_topo, sizes=[25, 10], device=0, mode='GPU')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_profiler_data_save_path(file_name: str) -> str:
    profiler_data_path = 'profiler/data/'
    if not osp.exists(profiler_data_path):
        os.mkdir(profiler_data_path)
    profiler_data_path = osp.join(profiler_data_path, file_name)
    return profiler_data_path


def get_nids_from_minibatch():
    """在单个GPU上采样得到的minibatch序列
    """
    file_path = get_profiler_data_save_path('nids.csv')
    if osp.exists(file_path):
        print("每个minibatch采样得到的节点已保存至", file_path)
        return
    i = 0
    df_list = []
    for seeds in train_loader:
        # seeds 为DataLoader迭代得到的minibatch
        i = i+1
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        n_id_list = n_id.cpu().numpy().tolist()
        temp_df = pd.DataFrame([n_id_list])
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True, axis=0)
    df.to_csv(file_path, index=False)


def get_nids_from_minibatch_parallel():
    """得到在两个GPU上分别采样到的minibatch节点序列
    """
    file_path_gpu0 = get_profiler_data_save_path('nids_gpu0.csv')
    file_path_gpu1 = get_profiler_data_save_path('nids_gpu1.csv')
    if osp.exists(file_path_gpu0) or osp.exists(file_path_gpu1):
        print("每个GPU的minibatch采样得到的节点已保存")
        return
    path_list = []
    path_list.append(file_path_gpu0)
    path_list.append(file_path_gpu1)
    train_loader_list = []
    train_loader_list.append(train_loader_gpu0)
    train_loader_list.append(train_loader_gpu1)

    for i in range(2):
        df_list = []
        for seeds in train_loader_list[i]:
            n_id, _, _ = quiver_sampler.sample(seeds)
            print("n_id:", n_id[:10])
            print("len:", len(n_id))
            n_id_list = n_id.cpu().numpy().tolist()
            temp_df = pd.DataFrame([n_id_list])
            df_list.append(temp_df)
        df = pd.concat(df_list, ignore_index=True, axis=0)
        df.to_csv(path_list[i], index=False)


def cache_feat_nums(cache_ratio: float, dataset: str) -> int:
    """根据数据集的节点特征数和设定的缓存比率得到缓存的特征数量

    Args:
        cache_ratio (float): 缓存比率，应该在0到1之间
        dataset (str): 数据集的名称

    Returns:
        int: 缓存特征的数量
    """
    assert cache_ratio >= 0 and cache_ratio <= 1, "\nCache ratio should between 0 and 1"
    feat_nums = 0
    if dataset == 'Reddit':
        feat_nums = 232965
    return int(feat_nums * cache_ratio)


def reindex_by_degree(adj_csr: quiver.CSRTopo, gpu_portion: float) -> Tuple[torch.Tensor, int]:
    """_summary_

    Args:
        adj_csr (quiver.CSRTopo): _description_
        gpu_portion (float): _description_

    Returns:
        _type_: _description_
    """
    node_count = adj_csr.indptr.shape[0]-1
    print("node number:", node_count)
    cache_nums = int(node_count*gpu_portion)
    perm_range = torch.randperm(cache_nums)
    print("cache size:", perm_range.size(0))
    # sort
    degree = adj_csr.indptr[1:]-adj_csr.indptr[:-1]
    print(degree)
    _, prev_order = torch.sort(degree, descending=True)
    # 将prev_order的前node_count*gpu_portion个元素打乱顺序
    prev_order[:cache_nums] = prev_order[perm_range]
    return prev_order, cache_nums


def get_feature_ids_on_two_gpus(adj_csr: quiver.CSRTopo, gpu_portion: float):
    print('='*20, 'BEGIN TO GENERATE', '='*20)
    processed_nids, cache_nums = reindex_by_degree(csr_topo, gpu_portion)
    block_size = cache_nums // 2
    feat_ids_gpu0 = processed_nids[:block_size].numpy().tolist()
    feat_ids_gpu1 = processed_nids[block_size:cache_nums].numpy().tolist()
    print(feat_ids_gpu0)
    print(feat_ids_gpu1)
    file_path = get_profiler_data_save_path(
        'cache_ids'+'_'+str(gpu_portion*100)+'%'+'.csv')
    df_list = []
    df_list.append(pd.DataFrame([feat_ids_gpu0]))
    df_list.append(pd.DataFrame([feat_ids_gpu1]))
    df = pd.concat(df_list, ignore_index=True, axis=0)
    df.to_csv(file_path, index=False)
    print('='*20, 'DONE', '='*20)


def hit_ratio_analysis_on_clique(gpu_portion: float):
    # 读取数据
    print("读取数据")
    cache_ids_path = get_profiler_data_save_path(
        'cache_ids'+'_'+str(gpu_portion*100)+'%'+'.csv')
    df0 = pd.read_csv(cache_ids_path)
    # 数据预处理
    print("数据预处理")
    df0.fillna(-1, inplace=True)
    cache_ids_gpu0 = df0.iloc[0].to_list()
    cache_ids_gpu1 = df0.iloc[1].to_list()
    cache_ids_gpu0 = list(map(int, cache_ids_gpu0))
    cache_ids_gpu1 = list(map(int, cache_ids_gpu1))
    cache_ids_gpu0 = [x for x in cache_ids_gpu0 if x != -1]
    cache_ids_gpu1 = [x for x in cache_ids_gpu1 if x != -1]
    cache_ids_gpu0_set = set(cache_ids_gpu0)
    cache_ids_gpu1_set = set(cache_ids_gpu1)

    # 创建DataFrame保存分析结果
    res_df = pd.DataFrame(
        columns=['gpu_id', 'batch_id', 'hit_ratio_local', 'hit_ratio_clique'])

    for i in range(2):
        batch_nids_gpui_path = get_profiler_data_save_path(
            'nids_gpu'+str(i)+'.csv')
        df = pd.read_csv(batch_nids_gpui_path)
        df.fillna(-1, inplace=True)
        rnums = df.shape[0]
        gpu_id = i  # tosheet
        print("GPU:", gpu_id)
        for j in range(rnums):
            minibatch_id = j  # tosheet
            batch_nids_gpu = df.iloc[j].to_list()
            batch_nids_gpu = list(map(int, batch_nids_gpu))
            batch_nids_gpu = [x for x in batch_nids_gpu if x != -1]
            batch_nids_gpu_set = set(batch_nids_gpu)
            batch_size = len(batch_nids_gpu_set)
            if i == 0:
                hit_local_count = len(cache_ids_gpu0_set & batch_nids_gpu_set)
                hit_clique_count = len(cache_ids_gpu1_set & batch_nids_gpu_set)
            elif i == 1:
                hit_local_count = len(cache_ids_gpu1_set & batch_nids_gpu_set)
                hit_clique_count = len(cache_ids_gpu0_set & batch_nids_gpu_set)
            else:
                hit_local_count = -1
                hit_clique_count = -1
            hit_local_ratio = hit_local_count / batch_size  # tosheet
            hit_clique_ratio = hit_clique_count / batch_size  # tosheet
            res_df.loc[i*rnums+j] = [int(gpu_id), int(minibatch_id),
                                     round(float(hit_local_ratio), 4), round(float(hit_clique_ratio), 4)]
    res_df['gpu_id'] = res_df['gpu_id'].astype(int)
    res_df['batch_id'] = res_df['batch_id'].astype(int)
    res_df['hit_ratio_clique'] = res_df['hit_ratio_clique'].apply(
        lambda x: '{:.2%}'.format(x))
    res_df['hit_ratio_local'] = res_df['hit_ratio_local'].apply(
        lambda x: '{:.2%}'.format(x))
    res_df['hit_ratio_clique'] = res_df['hit_ratio_clique'].astype(str)
    res_df['hit_ratio_local'] = res_df['hit_ratio_local'].astype(str)
    res_df.to_csv(get_profiler_data_save_path(
        'hit_analysis_'+str(gpu_portion*100)+'%.csv'), index=False)


def generate_cache_analysis(adj_csr: quiver.CSRTopo):
    for i in range(1, 10, 1):
        print(i/10)
        get_feature_ids_on_two_gpus(adj_csr, i/10)
        hit_ratio_analysis_on_clique(i/10)


if __name__ == '__main__':
    generate_cache_analysis(csr_topo)
