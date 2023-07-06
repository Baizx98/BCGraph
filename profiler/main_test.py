from general_test import CacheProfiler
import logging
import quiver
import torch
from torch_geometric.datasets import Reddit


def test_livejournel_hit():
    print("begin")
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="livejournal",
        gpu_list=[0],
        sample_gpu=0,
        static_cache_policy="degree",
        static_cache_ratio=0.2,
        dynamic_cache_policy="FIFO",
        dynamic_cache_ratio=0.000001,
        batch_size=6000,
    )
    cache.cache_nids_to_gpu()
    print("begin")

    (
        static_hit_ratio,
        dynamic_hit_ratio,
        relevant_dynamic_hit_ratio,
        global_hit_tatio,
    ) = cache.degree_and_fifo_mixed_analysis_on_single()
    print("static hit ration:{},cache ration:{:.2f}".format(static_hit_ratio, 0.2))


def test_livejournel_hit_static():
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="livejournal",
        gpu_list=[0],
        sample_gpu=0,
        static_cache_policy="degree",
        static_cache_ratio=0.2,
        batch_size=6000,
    )
    cache.cache_nids_to_gpu()
    logging.info("begin to compute")
    (
        hit_ratio,
        access_count,
        hit_count,
    ) = cache.static_cache_analysis_on_single()
    print(
        "hit ratio:{} \taccess count:{} \thit count:{}".format(
            hit_ratio, access_count, hit_count
        )
    )


def test_reddit_hit_static_frequency():
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="Reddit",
        gpu_list=[0],
        sample_gpu=0,
        static_cache_policy="frequency",
        static_cache_ratio=0.2,
        batch_size=1024,
    )
    cache.get_nids_all_frequency(4)
    cache.cache_nids_to_gpu()
    a = cache.static_cache_analysis_on_single()
    print(a[0])


def test_reddit_train_partition_msbfs_hit_static(cache_ratio: float = 0.8):
    # 测试reddit数据集在对训练集进行分区的情况下，使用预采样策略的命中率
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="Reddit",
        gpu_list=[0, 1, 2, 3],
        sample_gpu=0,
        static_cache_policy="frequency_separate",
        static_cache_ratio=cache_ratio,
        partition_strategy="msbfs",
    )
    cache.get_nids_all_frequency()
    cache.cache_nids_to_gpus()
    cache.static_cache_analysis_on_multiple()


def test_reddit_train_partition_linear_msbfs_hit_static(cache_ratio: float = 0.8):
    # 测试reddit数据集在对训练集进行基于多源BFS的线性划分的情况下，使用预采样策略的命中率
    logging.info("BEGIN")
    cache = CacheProfiler()
    cache.init_config(
        dataset_name="Reddit",
        gpu_list=[0, 1, 2, 3],
        sample_gpu=3,
        static_cache_policy="frequency_separate",
        static_cache_ratio=cache_ratio,
        partition_strategy="linear-msbfs",
    )
    cache.get_nids_all_frequency()
    cache.cache_nids_to_gpus()
    cache.static_cache_analysis_on_multiple()
    logging.info("DONE")


def test_sub_graph_num():
    dataset_path = "/home8t/bzx/data/Reddit"
    dataset = Reddit(dataset_path)
    data = dataset[0]
    train_idx: torch.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)
    idx = train_idx.numpy()
    edge_index: torch.Tensor = data.edge_index
    edge_list = edge_index.numpy().T
    csrtopo = quiver.CSRTopo(edge_index=edge_index)
    quiver_sample = quiver.pyg.GraphSageSampler(
        csr_topo=csrtopo, sizes=[25, 10], device=3, mode="GPU"
    )
    dataloader = torch.utils.data.DataLoader(
        train_idx,
        batch_size=1024,
        shuffle=True,
        drop_last=True,
    )
    for mini_batch in dataloader:
        n_id, _, _ = quiver_sample.sample(mini_batch)
        print(n_id.size(0))


if __name__ == "__main__":
    # for i in range(1, 10):
    #     print("第" + str(i) + "次")
    #     test_reddit_train_partition_linear_msbfs_hit_static(i / 10)
    test_reddit_hit_static_frequency()
