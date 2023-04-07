import os
import os.path as osp
import time
import logging
from typing import List

import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import quiver


class CacheProfiler:
    def __init__(self) -> None:
        # dataset
        self.dataset_name = None
        self.dataset_path = None
        self.dataset = None
        self.data = None
        self.train_idx = None
        self.train_idx_parallel = None
        self.csr_topo = None
        # environment
        self.world_size = 0
        self.gpu_list = []
        self.sample_gpu = 0
        self.cache_ratio = 0.5
        self.batch_size = 1024
        self.block_size = 0
        # analysis
        self.quiver_sample = None
        self.dataloader_list = []
        self.sample_nums_list = []
        self.gpu_frequency_list = []
        self.gpu_frequency_total: torch.Tensor = None
        self.gpu_cached_ids_list = []

        # log
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',  # 日志信息的格式化字符串
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.StreamHandler()])

        logging.info('CacheProfiler is created')

    def __str__(self) -> str:
        return self.dataset_name

    def init_config(self, dataset_name: str, gpu_list: List[int], cache_ratio: float, batch_size: int):
        # 初始化参数
        self.dataset_name = dataset_name
        self.dataset_path = self.get_dataset_save_path(dataset_name)
        self.gpu_list = gpu_list
        self.world_size = len(gpu_list)
        self.cache_ratio = cache_ratio
        self.batch_size = batch_size

        # 初始化数据集
        if self.dataset_name == 'Reddit':
            self.dataset = Reddit(self.dataset_path)
            self.data = self.dataset[0]
            self.train_idx: torch.Tensor = self.data.train_mask.nonzero(
                as_tuple=False).view(-1)
        if self.dataset_name == 'products':
            self.dataset = PygNodePropPredDataset(
                'ogbn-products', self.dataset_path)
            self.data = self.dataset[0]
            split_idx = self.dataset.get_idx_split()
            self.train_idx = split_idx['train']
        self.train_idx_parallel = self.train_idx.split(
            self.train_idx.size(0)//self.world_size)
        # 初始化DataLoader
        for i in range(self.world_size):
            train_loader_temp = torch.utils.data.DataLoader(self.train_idx_parallel[i],
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            drop_last=True)
            self.dataloader_list.append(train_loader_temp)
        # 初始化图拓扑结构
        self.csr_topo = quiver.CSRTopo(self.data.edge_index)
        # 初始化采样器
        self.quiver_sample = quiver.pyg.GraphSageSampler(
            self.csr_topo, sizes=[25, 10], device=self.sample_gpu, mode='GPU')

        # 初始化预采样全局频率统计
        self.gpu_frequency_total = torch.zeros(
            self.csr_topo.node_count, dtype=int)

        self.block_size = int(
            self.cache_ratio*self.csr_topo.node_count)//self.world_size
        logging.info('config inited')

    def reindex_by_degree(self):
        # 静态缓存 度排序
        degree = self.csr_topo.degree
        # 按度排序后得到的节点ID列表 prev_order
        _, prev_order = torch.sort(degree, descending=True)
        return prev_order

    def reindex_by_frequency_total(self) -> torch.Tensor:
        # 静态缓存 预采样频率排序 全局排序
        _, prev_order = torch.sort(
            self.gpu_frequency_total, descending=True)
        logging.info('reindex done')
        return prev_order

    def reindex_by_frequency_list(self):
        # 静态缓存 预采样频率排序 多GPU分别排序
        prev_order_list = []
        for li in self.gpu_frequency_list:
            _, temp = torch.sort(li, descending=True)
            prev_order_list.append(temp)
        logging.info('reindex done')
        return prev_order_list

    def cache_nids_to_gpus(self):
        # 将不同缓存策略下得到的缓存数据保存至不同的GPU缓存列表中
        # 多GPU各自预采样频率排序
        prev_order_list = self.reindex_by_frequency_list()
        for prev_order in prev_order_list:
            temp = prev_order[:self.block_size]
            self.gpu_cached_ids_list.append(temp)
        logging.info('cache done')
        return self.gpu_cached_ids_list

    def get_nids_all_frequency(self, epoch: int):

        if osp.exists(self.get_profiler_data_save_path(str(self.dataset_name)+'saved_frequency_data.pth')):
            saved_data = torch.load(self.get_profiler_data_save_path(
                str(self.dataset_name)+'saved_frequency_data.pth'))
            self.gpu_frequency_list = saved_data['gpu_frequency_list']
            self.gpu_frequency_total = saved_data['gpu_frequency_total']
            logging.info('read frequency data from file:' +
                         str(self.dataset_name))
        else:
            # 预采样 得到节点在不同GPU上的和全局的频率
            logging.info('begin pre-sample')
            for gpu_id in range(self.world_size):
                gpu_frequency_temp: torch.Tensor = torch.zeros(
                    self.csr_topo.node_count, dtype=int)
                for i in range(epoch):
                    print("epoch:", i)
                    for mini_batch in self.dataloader_list[gpu_id]:
                        n_id, _, _ = self.quiver_sample.sample(mini_batch)
                        gpu_frequency_temp[n_id] += 1
                        self.gpu_frequency_total[n_id] += 1
                self.gpu_frequency_list.append(gpu_frequency_temp)
            saved_data = {'gpu_frequency_list': self.gpu_frequency_list,
                          'gpu_frequency_total': self.gpu_frequency_total}
            torch.save(saved_data, self.get_profiler_data_save_path(
                str(self.dataset_name)+'saved_frequency_data.pth'))

        logging.info('pre-sample done')

    def cache_analysis_on_clique(self):
        """生成在两个GPU上的缓存命中率分析
        """
        logging.info('cache analysis begin')
        res_df = pd.DataFrame(
            columns=['gpu_id', 'batch_id', 'hit_ratio_local', 'hit_ratio_clique'])
        row_count = 0
        for i in range(2):
            logging.info('cache analysis on GPU'+str(i))
            for mini_batch_id, mini_batch in enumerate(self.dataloader_list[i]):
                n_id, _, _ = self.quiver_sample.sample(mini_batch)
                n_id_size = len(n_id)  # 采样得到的子图大小，其中节点无重复
                if i == 0:
                    hit_local_count = len(
                        set(self.gpu_cached_ids_list[i].tolist()) & set(n_id.tolist()))
                    hit_clique_count = len(
                        (set(self.gpu_cached_ids_list[1 - i].tolist())-set(self.gpu_cached_ids_list[i].tolist())) & set(n_id.tolist()))
                elif i == 1:
                    hit_local_count = len(
                        set(self.gpu_cached_ids_list[i].tolist()) & set(n_id.tolist()))
                    hit_clique_count = len(
                        (set(self.gpu_cached_ids_list[1-i].tolist())-set(self.gpu_cached_ids_list[i].tolist())) & set(n_id.tolist()))
                else:
                    hit_local_count = -1
                    hit_clique_count = -1
                hit_local_ratio = hit_local_count / n_id_size  # tosheet
                hit_clique_ratio = hit_clique_count / n_id_size  # tosheet
                res_df.loc[row_count] = [int(i), int(mini_batch_id),
                                         round(float(hit_local_ratio), 4), round(float(hit_clique_ratio), 4)]
                row_count += 1
        res_df['gpu_id'] = res_df['gpu_id'].astype(int)
        res_df['batch_id'] = res_df['batch_id'].astype(int)
        res_df['hit_ratio_clique'] = res_df['hit_ratio_clique'].apply(
            lambda x: '{:.2%}'.format(x))
        res_df['hit_ratio_local'] = res_df['hit_ratio_local'].apply(
            lambda x: '{:.2%}'.format(x))
        res_df['hit_ratio_clique'] = res_df['hit_ratio_clique'].astype(str)
        res_df['hit_ratio_local'] = res_df['hit_ratio_local'].astype(str)
        res_df.to_csv(self.get_profiler_data_save_path(
            str(self.dataset_name)+'_frequency_hit_analysis_'+str(self.cache_ratio*100)+'%.csv', profiler_data_path='profiler/data/frequency'), index=False)

        logging.info('cache analysis done')

    def get_dataset_save_path(self, dataset_name: str) -> str:
        return osp.join(osp.dirname(
            osp.realpath(__file__)), '..', 'data', dataset_name)

    def get_profiler_data_save_path(self, file_name: str, profiler_data_path='profiler/data/') -> str:
        """获取分析结果的保存路径

        Args:
            file_name (str): 文件名

        Returns:
            str: 分析结果的保存路径
        """
        if not osp.exists(profiler_data_path):
            os.mkdir(profiler_data_path)
        profiler_data_path = osp.join(profiler_data_path, file_name)
        return profiler_data_path


if __name__ == '__main__':
    for i in range(10):
        cache_temp = CacheProfiler()
        cache_temp.init_config(dataset_name='products', gpu_list=[
            0, 1], cache_ratio=(i+1)/10, batch_size=1024)
        cache_temp.get_nids_all_frequency(20)
        cache_temp.cache_nids_to_gpus()
        cache_temp.cache_analysis_on_clique()
