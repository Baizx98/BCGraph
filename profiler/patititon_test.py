import time
import random
import queue

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Reddit
import quiver

dataset_path = "/home8t/bzx/data/Reddit"
dataset = Reddit(dataset_path)
data = dataset[0]
train_idx: torch.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)
idx = train_idx.numpy()
edge_index: torch.Tensor = data.edge_index
edge_list = edge_index.numpy().T
print(edge_list.shape)
print(data.is_undirected())
print(idx)


def demo_test():
    G = nx.Graph()
    edges = [
        (0, 1, 2),
        (0, 2, 5),
        (0, 3, 1),
        (1, 2, 3),
        (1, 4, 1),
        (2, 3, 4),
        (2, 4, 3),
        (3, 5, 6),
        (4, 5, 2),
    ]
    G.add_weighted_edges_from(edges)
    print(nx.info(G))

    # 将部分节点加入最短路径计算集合
    start = time.time()
    nodes = [0, 1, 2, 4]
    shortest_path_length = {}
    for u in nodes:
        shortest_path_length[u] = {}
        for v in nodes:
            if u != v:
                shortest_path_length[u][v] = nx.shortest_path_length(G, u, v)

    print(shortest_path_length)
    end = time.time()
    print(end - start)
    # plt.savefig("demo.png")


def reddit_test():
    start = time.time()
    graph = nx.from_edgelist(edge_list, create_using=nx.Graph())
    # print(nx.info(graph))
    mid = time.time()
    print(mid - start)
    shortest_path_length = {}
    i = 0
    for u in idx:
        i += 1
        if i % 20 == 0:
            print(i)
        shortest_path_length[u] = {}
        for v in idx:
            if u != v:
                shortest_path_length[u][v] = nx.shortest_path_length(graph, u, v)
    print(shortest_path_length)
    end = time.time()
    print(str(end - mid))
    np.save("/home8t/bzx/padata/reddit/path.npy", shortest_path_length)


def train_idx_partition(
    csrtopo: quiver.CSRTopo,
    train_idx: np.ndarray,
    partition_num: int,
):
    # 创建与GPU数量相同的列表来保存为每个GPU分配的训练节点ID
    pa_list_dic = {i: [] for i in partition_num}

    train_idx_num = train_idx.shape[0]
    # accessed mask,用来标记训练节点是否被访问
    train_visited = [False] * train_idx_num
    full_visited = [False] * csrtopo.node_count

    train_visited_count = 0

    # 随机选取partition_num个起始节点,并加入队列作为初始值
    start_idx = random.sample(range(train_idx.shape[0]), partition_num)
    print(start_idx)
    start_idx = train_idx[start_idx]
    print(start_idx)

    # 创建各个分区的当前和下一轮的队列 open-closed表
    # now_que中初始化为种子节点
    bfs_queues = {
        i: {"now_que": queue.Queue(), "next_que": queue.Queue()}
        for i in range(partition_num)
    }

    while train_visited_count < train_idx_num:
        ...
        # 对pa_list中子list按照元素个数进行升序排序
        # sort(pa_list,sub_li_ele_num)
        # 得到排序后的列表 例如[4,2,1,3]
        pa_list_index_orderd = []
        for pa_id in pa_list_index_orderd:
            # 遍历队列中的节点，判断是否为train，并设置visited信息
            for nid in bfs_queues[pa_id]["now_que"]:  # 队列可以for循环吗？
                # TODO 先将当前遍历到的层中的节点的邻居节点（也就是下一层的节点）加入到下一层的队列当中
                ...
                #
                if full_visited[nid] == True:
                    continue
                full_visited[nid] = True
                if nid in train_idx:
                    if train_visited[nid] == True:
                        continue
                    pa_list_dic[pa_id].append(nid)
                    train_visited[nid] = True
                    train_visited_count += 1
            # 并且将每个节点的邻居节点添加到下一轮队列中
    ...


if __name__ == "__main__":
    # demo_test()
    # reddit_test()
    train_idx_partition(1, train_idx=idx, partition_num=4)
