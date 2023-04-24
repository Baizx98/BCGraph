from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import Reddit
import seaborn as sns
import multiprocessing as mp

import tool
from general_test import CacheProfiler, FIFOCache
from quiver import CSRTopo


class GraphConfig:
    img_save_path = "profiler/data/img/"

    def __init__(self) -> None:
        pass

    def img_path(file_name: str) -> str:
        return GraphConfig.img_save_path + file_name


def degree_distribution():
    """绘制节点度的直方图，描述不同度范围内节点的数量"""
    dataset = Reddit(tool.get_dataset_save_path("Reddit"))
    data = dataset[0]
    csr_topo = CSRTopo(data.edge_index)
    degree = csr_topo.degree
    order_degree, _ = torch.sort(degree)
    order_degree = order_degree.numpy()
    df = pd.DataFrame({"values": order_degree})

    # 使用 Seaborn 绘制直方图
    histogram = sns.histplot(data=df, x="values", bins=30000)

    # 显示直方图
    histogram.set(title="Histogram of Tensor data")
    plt.savefig(GraphConfig.img_save_path + "degree_distribution_Reddit" + ".png")


def degree_percent_distribution(dataset_name: str):
    print("beginnnn")
    if dataset_name == "Reddit":
        dataset = Reddit(tool.get_dataset_save_path(dataset_name))
    if dataset_name[:4] == "ogbn":
        print(tool.get_dataset_save_path())
        dataset = PygNodePropPredDataset(dataset_name, tool.get_dataset_save_path())
    data = dataset[0]
    csr_topo = CSRTopo(data.edge_index)
    degree = csr_topo.degree
    degree_sorted, _ = torch.sort(degree, descending=True)

    # 计算tensor总和
    total_sum = torch.sum(degree_sorted)
    # 计算占比的列表，包含每个位置的横坐标和纵坐标的值
    print("begin cal")
    x_values: torch.Tensor = (
        torch.arange(degree_sorted.shape[0]) / degree_sorted.shape[0]
    )
    y_values: torch.Tensor = torch.cumsum(degree_sorted, dim=0) / total_sum
    x = x_values.numpy()
    y = y_values.numpy()
    xx = x[::1000]
    yy = y[::1000]

    print("begin draw")
    # 使用Seaborn绘制散点图
    ax = sns.lineplot(x=xx, y=yy)

    # 设置横轴和纵轴标签
    ax.set(xlabel="Percentiles", ylabel="Cumulative sum percentage")
    # 显示绘图
    plt.savefig(
        tool.get_profiler_data_save_path(
            "degree_percent_distribution_" + dataset_name + ".png", "img"
        )
    )


def fifo_hit_ratio_contrast_run(rank, params):
    dataset_name, dynamic_cache_ratio, batch_size = params
    test = CacheProfiler()
    test.init_config(
        dataset_name=dataset_name,
        gpu_list=[0],
        sample_gpu=rank % 3,
        dynamic_cache_policy="FIFO",
        dynamic_cache_ratio=dynamic_cache_ratio,
        batch_size=batch_size,
    )
    hit_ratio = test.fifo_cache_hit_ratio_analysis_on_single()

    print(
        "dataset:"
        + dataset_name
        + " cache ratio:"
        + str(dynamic_cache_ratio)
        + " batch size:"
        + str(batch_size)
        + " hit ratio:"
        + str(hit_ratio)
    )
    return [dataset_name, dynamic_cache_ratio, batch_size, hit_ratio]


def fifo_hit_ratio_contrast():
    dataset = ["Reddit", "ogbn-products"]
    cache_ratio = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    batch_sizes = [256, 512, 1024, 2048]
    params_list = [
        (dataset_name, dynamic_cache_ratio, batch_size)
        for dataset_name in dataset
        for batch_size in batch_sizes
        for dynamic_cache_ratio in cache_ratio
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=3) as pool:
        res = pool.starmap(
            fifo_hit_ratio_contrast_run, [(r, p) for r, p in enumerate(params_list)]
        )
        res_df = pd.DataFrame(
            res, columns=["dataset", "cache_ratio", "batch_size", "hit_ratio"]
        )
        res_df.to_csv(
            tool.get_profiler_data_save_path(
                file_name="fifo_hit_ratio_analysis_on_single.csv", item="cache"
            ),
            index=False,
        )


def degree_and_fifo_ratio_contrast_run(rank, params):
    dataset_name, batch_size, static_cache_ratio, dynamic_cache_ratio = params
    test = CacheProfiler()
    test.init_config(
        dataset_name=dataset_name,
        gpu_list=[0],
        sample_gpu=rank % 2 + 1,
        static_cache_policy="degree",
        static_cache_ratio=static_cache_ratio,
        dynamic_cache_policy="FIFO",
        dynamic_cache_ratio=dynamic_cache_ratio,
        batch_size=batch_size,
    )
    test.cache_nids_to_gpu()
    (
        static_hit_ratio,
        dynamic_hit_ratio,
        relevant_dynamic_hit_ratio,
        global_hit_tatio,
    ) = test.degree_and_fifo_mixed_analysis_on_single()
    print(
        "dataset_name:",
        dataset_name,
        "batch_size:",
        batch_size,
        "static_cache_ratio:",
        static_cache_ratio,
        "dynamic_cache_ratio:",
        dynamic_cache_ratio,
        "static_hit_ratio:",
        static_hit_ratio,
        "dynamic_hit_ratio:",
        dynamic_hit_ratio,
        "relevant_dynamic_hit_ratio:",
        relevant_dynamic_hit_ratio,
        "global_hit_tatio:",
        global_hit_tatio,
    )
    return [
        dataset_name,
        batch_size,
        static_cache_ratio,
        dynamic_cache_ratio,
        static_hit_ratio,
        dynamic_hit_ratio,
        relevant_dynamic_hit_ratio,
        global_hit_tatio,
    ]


def degree_and_fifo_ratio_contrast():
    dataset = ["Reddit", "ogbn-products"]
    batch_sizes = [512, 1024]
    static_cache_ratio = [0.2, 0.3, 0.4]
    dynamic_cache_ratio = [0.1, 0.2, 0.3, 0.4]
    params_list = [
        (
            dataset_name,
            batch_size,
            static_ratio,
            dynamic_ratio,
        )
        for dataset_name in dataset
        for batch_size in batch_sizes
        for static_ratio in static_cache_ratio
        for dynamic_ratio in dynamic_cache_ratio
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=3) as pool:
        res = pool.starmap(
            degree_and_fifo_ratio_contrast_run,
            [(r, p) for r, p in enumerate(params_list)],
        )
        res_df = pd.DataFrame(
            res,
            columns=[
                "dataset_name",
                "batch_size",
                "static_cache_ratio",
                "dynamic_cache_ratio",
                "static_hit_ratio",
                "dynamic_hit_ratio",
                "relevant_dynamic_hit_ratio",
                "global_hit_tatio",
            ],
        )
        res_df.to_csv(
            tool.get_profiler_data_save_path(
                file_name="degree_and_fifo_ratio_contrast", item="cache"
            ),
            index=False,
        )


def fifo_hit_ratio_trendline():
    """FIFO策略早不同数据集、不同batch size下，缓存比例和命中率折线图对比"""
    df = pd.read_csv(
        tool.get_profiler_data_save_path(
            file_name="fifo_hit_ratio_analysis_on_single.csv", item="cache"
        )
    )
    print(df.columns)
    print(df.head)
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    ax = sns.lineplot(
        x="cache_ratio",
        y="hit_ratio",
        hue="dataset",
        style="batch_size",
        dashes=[(1, 1), (2, 1), (3, 2), (4, 1)],
        data=df,
    )
    ax.set(xlabel="cache ratio", ylabel="hit ratio")

    plt.savefig(tool.get_profiler_data_save_path("fifo_hit_ratio_trendline.png", "img"))


def fifo_test():
    access_list = [1, 2, 3] * 3
    cache = FIFOCache(3)
    hit_count = 0
    for id in access_list:
        if cache.get(id) == -1:
            cache.put(id, 1)
        else:
            hit_count += 1
    print(hit_count / len(access_list))


if __name__ == "__main__":
    # degree_percent_distribution("ogbn-papers100M")
    # fifo_hit_ratio_contrast()
    # cache_test()
    # fifo_hit_ratio_test()
    # fifo_test()
    # fifo_hit_ratio_trendline()
    degree_and_fifo_ratio_contrast()
