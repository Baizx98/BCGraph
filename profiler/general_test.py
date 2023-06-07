import os
import os.path as osp
import logging

import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

import pandas as pd
import numpy as np
import scipy.sparse as sp

import quiver
import tool


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
        self.batch_size = 1024
        # analysis
        self.quiver_sample = None
        self.dataloader = None
        self.dataloader_list = []
        self.sample_nums_list = []
        self.gpu_frequency_list = []
        self.gpu_frequency_total: torch.Tensor = None
        self.gpu_cached_ids = None  # 单个GPU缓存的节点id
        self.gpu_cached_ids_list: list[torch.Tensor] = []

        # cache
        self.static_cache_ratio = 0.5
        self.dynamic_cache_ratio = 0.2
        # static cache
        self.static_cache_policy = None
        self.static_cache_capacity = 0
        self.block_size = 0
        # dynamic cache
        self.dynamic_cache = None
        self.dynamic_cache_policy = None
        self.dynamic_cache_capacity = 0

        # log
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",  # 日志信息的格式化字符串
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
        logging.info("CacheProfiler is created")

    def __str__(self) -> str:
        return self.dataset_name

    def init_config(
        self,
        dataset_name: str,
        gpu_list: list[int],
        sample_gpu: int = 0,
        static_cache_policy: str = "",
        dynamic_cache_policy: str = "",
        static_cache_ratio: float = 0.0,
        dynamic_cache_ratio: float = 0.0,
        batch_size: int = 1024,
    ):
        # 初始化参数
        self.dataset_name = dataset_name
        self.dataset_path = tool.get_dataset_save_path()
        self.gpu_list = gpu_list
        self.world_size = len(gpu_list)
        self.static_cache_ratio = static_cache_ratio
        self.dynamic_cache_ratio = dynamic_cache_ratio
        self.batch_size = batch_size

        # 初始化数据集
        if self.dataset_name == "Reddit":
            self.dataset = Reddit(self.dataset_path + self.dataset_name)
            self.data = self.dataset[0]
            self.train_idx: torch.Tensor = self.data.train_mask.nonzero(
                as_tuple=False
            ).view(-1)
            # 初始化图拓扑结构
            self.csr_topo = quiver.CSRTopo(self.data.edge_index)
        # print(self.dataset_name[:4])
        if self.dataset_name[:4] == "ogbn":
            self.dataset = PygNodePropPredDataset(self.dataset_name, self.dataset_path)
            self.data = self.dataset[0]
            split_idx = self.dataset.get_idx_split()
            self.train_idx = split_idx["train"]
            # 初始化图拓扑结构
            self.csr_topo = quiver.CSRTopo(self.data.edge_index)

        if self.dataset_name == "livejournal":
            edge_index = torch.load(
                osp.join(self.dataset_path, self.dataset_name, "edge_index.pt")
            )
            self.csr_topo = quiver.CSRTopo(edge_index)
            train_mask = np.load(
                osp.join(self.dataset_path, self.dataset_name, "train.npy")
            )
            train_id = np.nonzero(train_mask)[0].astype(np.int64)
            self.train_idx = torch.from_numpy(train_id)

        if self.dataset_name == "sub_Reddit":
            sub_dataset_path = "/home8t/bzx/padata/reddit/4naive/"
            sub_train_id = np.load(os.path.join(sub_dataset_path, "sub_trainid_0.npy"))
            sub_adj = sp.load_npz(os.path.join(sub_dataset_path, "subadj_0.npz"))
            self.train_idx = torch.from_numpy(sub_train_id)
            self.csr_topo = quiver.CSRTopo(
                indptr=sub_adj.indptr, indices=sub_adj.indices
            )

        if self.dataset_name == "sub_ogbn-products":
            sub_dataset_path = "/home8t/bzx/padata/ogbn_products/4naive/"
            self.train_idx = torch.from_numpy(
                np.load(os.path.join(sub_dataset_path, "sub_trainid_0.npy"))
            )
            sub_adj = sp.load_npz(os.path.join(sub_dataset_path, "subadj_0.npz"))
            self.csr_topo = quiver.CSRTopo(
                indptr=sub_adj.indptr, indices=sub_adj.indices
            )
        logging.info("node count:" + str(self.csr_topo.node_count))
        # 初始化DataLoader
        self.dataloader = torch.utils.data.DataLoader(
            self.train_idx,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        if self.world_size > 1:
            self.train_idx_parallel = self.train_idx.split(
                self.train_idx.size(0) // self.world_size
            )
            for i in range(self.world_size):
                train_loader_temp = torch.utils.data.DataLoader(
                    self.train_idx_parallel[i],
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                self.dataloader_list.append(train_loader_temp)
            self.block_size = (
                int(self.static_cache_ratio * self.csr_topo.node_count)
                // self.world_size
            )

        # 初始化采样器
        self.sample_gpu = sample_gpu
        self.quiver_sample = quiver.pyg.GraphSageSampler(
            self.csr_topo, sizes=[25, 10], device=self.sample_gpu, mode="GPU"
        )

        # 初始化缓存策略
        self.static_cache_policy = static_cache_policy
        self.static_cache_capacity = int(self.csr_topo.node_count * static_cache_ratio)
        logging.info("sata capacity:" + str(self.static_cache_capacity))
        self.dynamic_cache_policy = dynamic_cache_policy
        self.dynamic_cache_capacity = int(
            self.csr_topo.node_count * dynamic_cache_ratio
        )
        logging.info("dy capacity:" + str(self.dynamic_cache_capacity))

        if dynamic_cache_policy == "FIFO":
            logging.info("初始化FIFO")
            self.dynamic_cache = FIFOCache(self.dynamic_cache_capacity)

        # 初始化预采样全局频率统计
        self.gpu_frequency_total = torch.zeros(self.csr_topo.node_count, dtype=int)

        logging.info("config inited")

    def cache_nids_to_gpus(self):
        # 静态缓存 将不同缓存策略下得到的缓存数据保存至不同的GPU缓存列表中

        # 多GPU各自预采样频率排序
        if self.static_cache_policy == "frequency_separate":
            prev_order_list = tool.reindex_nid_by_hot_metric(self.gpu_frequency_list)
            for prev_order in prev_order_list:
                temp = prev_order[: self.block_size]
                self.gpu_cached_ids_list.append(temp)
        logging.info("cache done")
        if self.static_cache_policy == "degree":
            ...
        if self.static_cache_policy == "batch_replace":
            ...
        return

    def get_nids_all_frequency(self, epoch: int):
        """预采样获取节点访问频率

        Args:
            epoch (int): 预采样的轮数
        """
        data_path = self.dataset_name + "_" + str(epoch) + "_saved_frequency_data.pth"
        if osp.exists(tool.get_profiler_data_save_path(data_path)):
            saved_data = torch.load(tool.get_profiler_data_save_path(data_path))
            self.gpu_frequency_list = saved_data["gpu_frequency_list"]
            self.gpu_frequency_total = saved_data["gpu_frequency_total"]
            logging.info("read frequency data from file:" + str(self.dataset_name))
        else:
            # 预采样 得到节点在不同GPU上的和全局的频率
            logging.info("begin pre-sample")
            for gpu_id in range(self.world_size):
                gpu_frequency_temp: torch.Tensor = torch.zeros(
                    self.csr_topo.node_count, dtype=int
                )
                for i in range(epoch):
                    if i % 5 == 0:
                        print("epoch:", i)
                    for mini_batch in self.dataloader_list[gpu_id]:
                        n_id, _, _ = self.quiver_sample.sample(mini_batch)
                        gpu_frequency_temp[n_id.cpu()] += 1
                        self.gpu_frequency_total[n_id.cpu()] += 1
                self.gpu_frequency_list.append(gpu_frequency_temp)
            saved_data = {
                "gpu_frequency_list": self.gpu_frequency_list,
                "gpu_frequency_total": self.gpu_frequency_total,
            }
            torch.save(
                saved_data,
                tool.get_profiler_data_save_path(data_path),
            )

        logging.info("{} epoch pre-sample done".format(epoch))

    def get_comprehensive_score_of_multiple_indicators(self):
        """生成节点重要度的综合衡量评价指标"""
        pass

    def cache_analysis_on_clique(self):
        """生成在两个GPU上的缓存命中率分析"""
        logging.info("cache analysis begin")
        res_df = pd.DataFrame(
            columns=["gpu_id", "batch_id", "hit_ratio_local", "hit_ratio_clique"]
        )
        row_count = 0
        for i in range(2):
            logging.info("cache analysis on GPU" + str(i))
            for mini_batch_id, mini_batch in enumerate(self.dataloader_list[i]):
                n_id, _, _ = self.quiver_sample.sample(mini_batch)
                n_id_size = len(n_id)  # 采样得到的子图大小，其中节点无重复
                if i == 0:
                    hit_local_count = len(
                        set(self.gpu_cached_ids_list[i].tolist()) & set(n_id.tolist())
                    )
                    hit_clique_count = len(
                        (
                            set(self.gpu_cached_ids_list[1 - i].tolist())
                            - set(self.gpu_cached_ids_list[i].tolist())
                        )
                        & set(n_id.tolist())
                    )
                elif i == 1:
                    hit_local_count = len(
                        set(self.gpu_cached_ids_list[i].tolist()) & set(n_id.tolist())
                    )
                    hit_clique_count = len(
                        (
                            set(self.gpu_cached_ids_list[1 - i].tolist())
                            - set(self.gpu_cached_ids_list[i].tolist())
                        )
                        & set(n_id.tolist())
                    )
                else:
                    hit_local_count = -1
                    hit_clique_count = -1
                hit_local_ratio = hit_local_count / n_id_size  # tosheet
                hit_clique_ratio = hit_clique_count / n_id_size  # tosheet
                res_df.loc[row_count] = [
                    int(i),
                    int(mini_batch_id),
                    round(float(hit_local_ratio), 4),
                    round(float(hit_clique_ratio), 4),
                ]
                row_count += 1
        res_df["gpu_id"] = res_df["gpu_id"].astype(int)
        res_df["batch_id"] = res_df["batch_id"].astype(int)
        res_df["hit_ratio_clique"] = res_df["hit_ratio_clique"].apply(
            lambda x: "{:.2%}".format(x)
        )
        res_df["hit_ratio_local"] = res_df["hit_ratio_local"].apply(
            lambda x: "{:.2%}".format(x)
        )
        res_df["hit_ratio_clique"] = res_df["hit_ratio_clique"].astype(str)
        res_df["hit_ratio_local"] = res_df["hit_ratio_local"].astype(str)
        res_df.to_csv(
            tool.get_profiler_data_save_path(
                str(self.dataset_name)
                + "_hit_analysis_"
                + str(self.batch_size)
                + "_"
                + str(self.static_cache_ratio * 100)
                + "%.csv",
                profiler_data_path="profiler/data/" + self.static_cache_policy,
            ),
            index=False,
        )

        logging.info("cache analysis done")

    def cache_repetition_analysis_on_clique(self):
        """统计缓存在两个GPU上的节点之间的重复率"""

        if not self.gpu_cached_ids_list:
            logging.info("Data not cached")
        print("GPU0 cached id nums: ", self.gpu_cached_ids_list[0].shape[0])
        print("GPU1 cached id nums: ", self.gpu_cached_ids_list[1].shape[0])
        print(
            "Duplicate id nums: ",
            len(
                set(self.gpu_cached_ids_list[0].tolist())
                & set(self.gpu_cached_ids_list[1].tolist())
            ),
        )
        print(
            "Dupliacate ratio:",
            len(
                set(self.gpu_cached_ids_list[0].tolist())
                & set(self.gpu_cached_ids_list[1].tolist())
            )
            / self.gpu_cached_ids_list[0].shape[0],
        )

    def batch_repetition_analysis_on_clique(self):
        """统计两个GPU上分别采样得到的batch之间的重复数量"""

        res_df = pd.DataFrame(
            columns=[
                "batch_id",
                "gpu0_subgraph_num",
                "gpu1_subgraph_num",
                "subgraph_repetition_num",
                "repetition_0_rate",
                "repetition_1_rate",
            ]
        )

        for mini_batch_id, (mini_batch_gpu0, mini_batch_gpu1) in enumerate(
            zip(self.dataloader_list[0], self.dataloader_list[1])
        ):
            n_id_gpu0, _, _ = self.quiver_sample.sample(mini_batch_gpu0)
            n_id_gpu1, _, _ = self.quiver_sample.sample(mini_batch_gpu1)

            batch_id = mini_batch_id
            gpu0_subgraph_num = n_id_gpu0.shape[0]
            gpu1_subgraph_num = n_id_gpu1.shape[0]
            subgraph_repetition_num = len(
                set(n_id_gpu0.tolist()) & set(n_id_gpu1.tolist())
            )  # 在两个GPU上分别采样得到的子图之间重复的节点数量
            repetition_0_rate = subgraph_repetition_num / gpu0_subgraph_num
            repetition_1_rate = subgraph_repetition_num / gpu1_subgraph_num

            res_df.loc[mini_batch_id] = [
                batch_id,
                gpu0_subgraph_num,
                gpu1_subgraph_num,
                subgraph_repetition_num,
                repetition_0_rate,
                repetition_1_rate,
            ]
        res_df["batch_id"] = res_df["batch_id"].astype(int)
        res_df["gpu0_subgraph_num"] = res_df["gpu0_subgraph_num"].astype(int)
        res_df["gpu1_subgraph_num"] = res_df["gpu1_subgraph_num"].astype(int)
        res_df["subgraph_repetition_num"] = res_df["subgraph_repetition_num"].astype(
            int
        )
        res_df["repetition_0_rate"] = res_df["repetition_0_rate"].apply(
            lambda x: "{:.2%}".format(x)
        )
        res_df["repetition_1_rate"] = res_df["repetition_1_rate"].apply(
            lambda x: "{:.2%}".format(x)
        )
        res_df["repetition_0_rate"] = res_df["repetition_0_rate"].astype(str)
        res_df["repetition_1_rate"] = res_df["repetition_1_rate"].astype(str)
        res_df.to_csv(
            tool.get_profiler_data_save_path(
                str(self.dataset_name)
                + "_batch_analysis_"
                + str(self.batch_size)
                + "_"
                + str(self.static_cache_ratio * 100)
                + "%.csv",
                profiler_data_path="profiler/data/" + "batch",
            ),
            index=False,
        )
        logging.info("batch repetiton analysis done")

    def fifo_cache_hit_ratio_analysis_on_single(self, pre_cache=False):
        if pre_cache and self.dynamic_cache_capacity > 0:
            logging.info("pre cache with degree policy")
            order = torch.sort(self.csr_topo.degree, descending=True)
            for id in order[
                self.static_cache_capacity : self.static_cache_capacity
                + self.dynamic_cache_capacity
            ]:
                id = int(id.tolist())
                self.dynamic_cache.put(id, 1)
        hit_count = 0
        access_count = 0
        logging.info(self.dataset_name + " begin test")
        for mini_batch in self.dataloader:
            n_id, _, _ = self.quiver_sample.sample(mini_batch)
            access_count += len(n_id)
            for id in n_id:
                id = int(id.tolist())
                if self.dynamic_cache.get(id) == -1:  # 未命中
                    self.dynamic_cache.put(id, 1)
                else:  # 命中
                    hit_count += 1
        hit_ratio = hit_count / access_count
        # 数据集、动态缓存比例、batch size
        return hit_ratio

    def cache_nids_to_gpu(self):
        """单GPU模式下将节点缓存至GPU中"""
        if self.static_cache_policy == "degree":
            degree = self.csr_topo.degree
            _, degree_orderd = torch.sort(degree, descending=True)
            self.gpu_cached_ids = degree_orderd[: self.static_cache_capacity]
        if self.static_cache_policy == "frequency":
            self.get_nids_all_frequency(3)
            _, frequency_orderd = torch.sort(self.gpu_frequency_total, descending=True)
            self.gpu_cached_ids = frequency_orderd[: self.static_cache_capacity]
        logging.info("cached done")

    def cache_nids_to_gpu_nvlink(self):
        """模拟在4个GPU NVLink互连的情况下，使用degree排序策略将排名靠前的节点特征交叉缓存至每个GPU后
        GPU0上的缓存情况
        """
        if self.static_cache_policy == "degree":
            _, degree_orderd = torch.sort(self.csr_topo.degree, descending=True)
            self.gpu_cached_ids = degree_orderd[
                : self.static_cache_capacity : self.world_size
            ]
            print(degree_orderd[0 : self.static_cache_capacity : self.world_size])
            logging.info("nvlink cache done")

    def degree_and_fifo_mixed_analysis_on_single(self, pre_cache=False):
        static_hit_count = 0
        dynamic_hit_count = 0
        access_count = 0
        dynamic_access_count = 0
        batch_count = 0
        if pre_cache:
            logging.info("pre cache with degree policy")
            order = torch.sort(self.csr_topo.degree, descending=True)
            for id in order[
                self.static_cache_capacity : self.static_cache_capacity
                + self.dynamic_cache_capacity
            ]:
                id = int(id.tolist())
                self.dynamic_cache.put(id, 1)
        my_dataloader = None
        if self.world_size > 1:
            my_dataloader = self.dataloader_list[0]
        else:
            my_dataloader = self.dataloader
        for mini_batch in my_dataloader:
            n_id, _, _ = self.quiver_sample.sample(mini_batch)
            access_count += len(n_id)
            batch_count += 1
            static_hit_nids: set = set(self.gpu_cached_ids.tolist()) & set(
                n_id.tolist()
            )
            static_hit_count += len(static_hit_nids)
            dynamic_access_nids: set = set(n_id.tolist()) - static_hit_nids
            dynamic_access_count += len(dynamic_access_nids)
            dynamic_access_nids = list(dynamic_access_nids)
            n_id = n_id.tolist()
            dynamic_access_nids = [x for x in n_id if x not in static_hit_nids]
            if batch_count % 10 == 0:
                ...
                # print("batch ratio:", len(static_hit_nids) / len(n_id))
            for id in dynamic_access_nids:
                if self.dynamic_cache.get(id) == -1:
                    self.dynamic_cache.put(id, 1)
                else:
                    # print("hit!")
                    dynamic_hit_count += 1
        logging.info("access node num:" + str(int(access_count / batch_count)))
        static_hit_ratio = static_hit_count / access_count
        dynamic_hit_ratio = dynamic_hit_count / access_count
        if dynamic_access_count != 0:
            relevant_dynamic_hit_ratio = dynamic_hit_count / dynamic_access_count
        else:
            relevant_dynamic_hit_ratio = 0
        global_hit_tatio = (static_hit_count + dynamic_hit_count) / access_count
        return (
            static_hit_ratio,
            dynamic_hit_ratio,
            relevant_dynamic_hit_ratio,
            global_hit_tatio,
        )

    def static_cache_analysis_on_single(self):
        hit_count = 0
        access_count = 0
        for batch_id, mini_batch in enumerate(self.dataloader):
            n_id, _, _ = self.quiver_sample.sample(mini_batch)
            access_count += n_id.size(0)
            hit_n_id: set = set(self.gpu_cached_ids.tolist()) & set(n_id.tolist())
            hit_count += len(hit_n_id)
            if batch_id % 10 == 0:
                print(
                    "{}".format(
                        n_id.size(0),
                    )
                )
        hit_ratio = hit_count / access_count
        return (
            hit_ratio,
            access_count,
            hit_count,
        )


class FIFOCache:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.cache = {}
        self.list = []

    def get(self, nid):
        if nid in self.cache:
            return self.cache[nid]
        else:
            return -1

    def put(self, nid: int, feature):
        # 模拟阶段，所有节点特征用1来表示
        if nid in self.cache:
            self.cache[nid] = feature
        else:
            if self.capacity == 0:
                return
            if len(self.list) == self.capacity:
                oldest_nid = self.list.pop(0)
                self.cache.pop(oldest_nid)
            self.list.append(nid)
            self.cache[nid] = feature


if __name__ == "__main__":
    # 生成不同缓存比例下采用频率热度排序后在clique内的缓存命中率
    # for i in range(10):
    #     cache_temp = CacheProfiler()
    #     cache_temp.init_config(
    #         dataset_name="ogbn-products",
    #         gpu_list=[0, 1],
    #         static_cache_policy="frequency_separate",
    #         static_cache_ratio=(i + 1) / 10,
    #         batch_size=1024,
    #     )
    #     cache_temp.get_nids_all_frequency(20)
    #     cache_temp.cache_nids_to_gpus()
    #     cache_temp.cache_analysis_on_clique()

    # cache_batch = CacheProfiler()
    # cache_batch.init_config(
    #     dataset_name="products",
    #     gpu_list=[0, 1],
    #     cache_policy=None,
    #     cache_ratio=0,
    #     batch_size=2048,
    # )
    # cache_batch.batch_repetition_analysis_on_clique()
    test = CacheProfiler()
    test.init_config(
        dataset_name="sub_ogbn-products",
        gpu_list=[0, 1, 2, 3],
        sample_gpu=1,
        static_cache_policy="degree",
        static_cache_ratio=0.2,
        dynamic_cache_policy="FIFO",
        dynamic_cache_ratio=0,
        batch_size=1024,
    )
    if test.dataset_name[:3] == "sub":
        test.cache_nids_to_gpu()
    else:
        test.cache_nids_to_gpu_nvlink()
    (
        static_hit_ratio,
        dynamic_hit_ratio,
        relevant_dynamic_hit_ratio,
        global_hit_tatio,
    ) = test.degree_and_fifo_mixed_analysis_on_single()
    print(
        "static_hit_ratio:",
        static_hit_ratio,
        "dynamic_hit_ratio:",
        dynamic_hit_ratio,
        "relevant_dynamic_hit_ratio:",
        relevant_dynamic_hit_ratio,
        "global_hit_tatio:",
        global_hit_tatio,
    )
    print("static hit ration:{},cache ration:{:.2f}".format(static_hit_ratio, 0.223695))

    # for i in range(10):
    #     test.init_config(
    #         dataset_name="sub_ogbn-products",
    #         gpu_list=[0, 1, 2, 3],
    #         sample_gpu=1,
    #         static_cache_policy="degree",
    #         static_cache_ratio=(i + 1) / 10,
    #         dynamic_cache_policy="FIFO",
    #         dynamic_cache_ratio=0,
    #         batch_size=1024,
    #     )
    #     if test.dataset_name[:3] == "sub":
    #         test.cache_nids_to_gpu()
    #     else:
    #         test.cache_nids_to_gpu_nvlink()
    #     (
    #         static_hit_ratio,
    #         dynamic_hit_ratio,
    #         relevant_dynamic_hit_ratio,
    #         global_hit_tatio,
    #     ) = test.degree_and_fifo_mixed_analysis_on_single()
    #     # print(
    #     #     "static_hit_ratio:",
    #     #     static_hit_ratio,
    #     #     "dynamic_hit_ratio:",
    #     #     dynamic_hit_ratio,
    #     #     "relevant_dynamic_hit_ratio:",
    #     #     relevant_dynamic_hit_ratio,
    #     #     "global_hit_tatio:",
    #     #     global_hit_tatio,
    #     # )
    #     print(
    #         "static hit ration:{},cache ration:{:.2f}".format(
    #             static_hit_ratio, (i + 1) / 10
    #         )
    #     )
