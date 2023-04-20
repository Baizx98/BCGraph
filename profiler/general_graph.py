import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import Reddit

import tool
from quiver import CSRTopo


class GraphConfig:
    img_save_path = "/profiler/data/img/"

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

    #    使用 Seaborn 绘制直方图
    histogram = sns.histplot(data=df, x="values", bins=30000)

    # 显示直方图
    histogram.set(title="Histogram of Tensor data")
    plt.savefig("profiler/data/" + "img/" + "degree_distribution_Reddit" + ".png")


def degree_percent_distribution():
    print("begin")
    dataset = Reddit(tool.get_dataset_save_path("Reddit"))
    data = dataset[0]
    csr_topo = CSRTopo(data.edge_index)
    degree = csr_topo.degree
    degree_sorted, _ = torch.sort(degree, descending=True)

    # 计算tensor总和
    total_sum = torch.sum(degree_sorted)
    # 计算占比的列表，包含每个位置的横坐标和纵坐标的值
    x_values: torch.Tensor = (
        torch.arange(degree_sorted.shape[0]) / degree_sorted.shape[0]
    )
    y_values: torch.Tensor = torch.cumsum(degree_sorted, dim=0) / total_sum
    x = x_values.numpy()
    y = y_values.numpy()

    print("begin draw")
    # 使用Seaborn绘制散点图
    ax = sns.lineplot(x=x, y=y)

    # 设置横轴和纵轴标签
    ax.set(xlabel="Percentiles", ylabel="Cumulative sum percentage")
    # 显示绘图
    plt.savefig(
        tool.get_profiler_data_save_path(
            "degree_percent_distribution_Reddit.png", "img"
        )
    )


if __name__ == "__main__":
    print("hello")
    degree_percent_distribution()
