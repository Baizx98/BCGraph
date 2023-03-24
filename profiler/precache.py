import os.path as osp
import os
import time
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data

import quiver
import pandas as pd

import sampler as myutil

gpu_count = 2

dataset_path = osp.join(osp.dirname(
    osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(dataset_path)
data: Data = dataset[0]
train_idx: torch.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)
train_idx_parallel = train_idx.split(train_idx.size(0)//gpu_count)
csr_topo = quiver.CSRTopo(data.edge_index)
# print(csr_topo.indptr.shape[0])
# print(data.edge_index.shape)
# print(csr_topo.indices)
# print(csr_topo)

train_loader_gpui_list = []
for i in range(gpu_count):
    train_loader_gpui_list.append(torch.utils.data.DataLoader(train_idx_parallel[i],
                                                              batch_size=1024,
                                                              shuffle=True,
                                                              drop_last=True))

gpu_weight = torch.zeros(232965, dtype=int)

print("gpu weight :::", gpu_weight)


def get_neighbors(minibatch, gpu_weight):

    # 计算邻居节点
    start_idx = csr_topo.indptr[minibatch]
    end_idx = csr_topo.indptr[minibatch + 1]
    print("start:", start_idx.shape[0])
    print("end:", end_idx.shape[0])
    print("gpu_weight:", gpu_weight.shape)

    for s, e in zip(start_idx, end_idx):
        neighbor_idx = csr_topo.indices[s:e]
        for id in neighbor_idx:
            gpu_weight[id] = gpu_weight[id]+1
    return neighbor_idx


def get_gpu_weight(train_loader_gpui_list: list):
    """保存dataloader某次加载数据后得到的特征在不同GPU上的保存权重

    Args:
        train_loader_gpui_list (list): 在不同GPU上运行的minibatch加载器
    """
    file_path = myutil.get_profiler_data_save_path('gpu_weight.csv')
    if osp.exists(file_path):
        print("file has been generated")
        return

    gpu_weight_list = []
    for j in range(gpu_count):
        x = torch.zeros(232965, dtype=int)
        gpu_weight_list.append(x)
    for i in range(gpu_count):
        for minibatch in train_loader_gpui_list[i]:
            start_idx = csr_topo.indptr[minibatch]
            end_idx = csr_topo.indptr[minibatch+1]
            print(time.time())
            for s, e in zip(start_idx, end_idx):
                neighbors_idx = csr_topo.indices[s:e]
                for id in neighbors_idx:
                    gpu_weight_list[i][id] = gpu_weight_list[i][id]+1
    df = []
    for i in range(gpu_count):
        gpu_weight = gpu_weight_list[i]
        gpu_weight = gpu_weight.numpy().tolist()
        temp_df = pd.DataFrame([gpu_weight])
        df.append(temp_df)
    df = pd.concat(df, ignore_index=True, axis=0)
    df.to_csv(file_path, index=False)


def reindex_by_degree_only(adj_csr: quiver.CSRTopo, gpu_portion: float):
    """将节点id按度排序，并且将前x%的节点结合权重分配给不同的GPU

    Args:
        ajd_csr (quiver.CSRTopo): _description_
        gpu_portion (float): _description_
    """
    node_count = adj_csr.indptr.shape[0]-1
    print("node number:", node_count)
    cache_nums = int(node_count*gpu_portion)
    # sort
    degree = adj_csr.indptr[1:]-adj_csr.indptr[:-1]
    print(degree)
    # 按度排序后得到的节点ID列表 prev_order
    _, prev_order = torch.sort(degree, descending=True)
    return prev_order, cache_nums


def get_feature_ids_on_two_gpus_with_weight(adj_csr: quiver.CSRTopo, gpu_portion: float):
    file_path = myutil.get_profiler_data_save_path(
        'weight_cache_ids'+'_'+str(gpu_portion*100)+'%'+'.csv')
    if osp.exists(file_path):
        print('文件已生成')
        return
    print('='*20, gpu_portion*100, '%', 'BEGIN', '='*20)
    pre_processed_nids, cache_nums = reindex_by_degree_only(
        adj_csr, gpu_portion)
    block_size = cache_nums//gpu_count
    cache_ids = pre_processed_nids[:cache_nums]
    gpu_weight_path = myutil.get_profiler_data_save_path('gpu_weight.csv')
    df = pd.read_csv(gpu_weight_path)
    gpu0_weight = df.iloc[0].to_list()
    gpu1_weight = df.iloc[1].to_list()
    # print(pre_processed_nids)
    gpu0_cache_ids = []
    gpu1_cache_ids = []
    weight_bool = [gpu0_weight[i]-gpu1_weight[i]
                   for i in range(len(gpu0_weight))]
    gpu0_indices = [i for i, x in enumerate(weight_bool) if x > 0]
    gpu1_indices = [i for i, x in enumerate(weight_bool) if x < 0]
    print("gpu0 nums:", len(gpu0_indices))
    print("gpu1 nums:", len(gpu1_indices))
    for i in range(cache_nums):
        nid = int(pre_processed_nids[i])
        if len(gpu0_cache_ids) >= block_size:
            gpu1_cache_ids.append(nid)
            continue
        if len(gpu1_cache_ids) >= block_size:
            gpu0_cache_ids.append(nid)
            continue

        if gpu0_weight[nid] > gpu1_weight[nid]:
            gpu0_cache_ids.append(nid)
        else:
            gpu1_cache_ids.append(nid)
    print("gpu0 cache id nums:", len(gpu0_cache_ids))
    print("gpu1 cache id nums:", len(gpu1_cache_ids))

    df_list = []
    df_list.append(pd.DataFrame([gpu0_cache_ids]))
    df_list.append(pd.DataFrame([gpu1_cache_ids]))
    res_df = pd.concat(df_list, ignore_index=True, axis=0)
    res_df.to_csv(file_path, index=False)
    print('='*20, gpu_portion*100, '%', 'END', '='*20)


def hit_ratio_analysis_on_clique_with_weight(gpu_portion: float):
    print("读取数据")
    cache_ids_path = myutil.get_profiler_data_save_path(
        'weight_cache_ids'+'_'+str(gpu_portion*100)+'%'+'.csv')
    df0 = pd.read_csv(cache_ids_path)

    # 数据预处理
    print("数据预处理")
    df0.fillna(-1, inplace=True)
    weight_cache_ids_gpu0 = df0.iloc[0].to_list()
    weight_cache_ids_gpu1 = df0.iloc[1].to_list()
    weight_cache_ids_gpu0 = list(map(int, weight_cache_ids_gpu0))
    weight_cache_ids_gpu1 = list(map(int, weight_cache_ids_gpu1))
    weight_cache_ids_gpu0 = [x for x in weight_cache_ids_gpu0 if x != -1]
    weight_cache_ids_gpu1 = [x for x in weight_cache_ids_gpu1 if x != -1]
    weight_cache_ids_gpu0_set = set(weight_cache_ids_gpu0)
    weight_cache_ids_gpu1_set = set(weight_cache_ids_gpu1)
    # 创建DataFrame保存分析结果
    res_df = pd.DataFrame(
        columns=['gpu_id', 'batch_id', 'hit_ratio_local', 'hit_ratio_clique'])

    for i in range(2):
        batch_nids_gpui_path = myutil.get_profiler_data_save_path(
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
                hit_local_count = len(
                    weight_cache_ids_gpu0_set & batch_nids_gpu_set)
                hit_clique_count = len(
                    weight_cache_ids_gpu1_set & batch_nids_gpu_set)
            elif i == 1:
                hit_local_count = len(
                    weight_cache_ids_gpu1_set & batch_nids_gpu_set)
                hit_clique_count = len(
                    weight_cache_ids_gpu0_set & batch_nids_gpu_set)
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
    res_df.to_csv(myutil.get_profiler_data_save_path(
        'weight_hit_analysis_'+str(gpu_portion*100)+'%.csv'), index=False)
    print("分析完毕")


def generate_cache_analysis_with_weight(adj_csr: quiver.CSRTopo):
    for i in range(1, 10, 1):
        print(i/10)
        get_feature_ids_on_two_gpus_with_weight(adj_csr, i/10)
        hit_ratio_analysis_on_clique_with_weight(i/10)


if __name__ == '__main__':
    generate_cache_analysis_with_weight(csr_topo)
