import time
import random
import queue
import collections
import os.path as osp

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


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    print("time to build graph:", mid - start)
    print(type(train_idx))
    print(train_idx)
    source_nodes = random.sample(train_idx.tolist(), 4)
    print(source_nodes)
    spl_dic = {}
    for i, start_node in enumerate(source_nodes):
        shortest_path = nx.single_source_dijkstra_path_length(graph, start_node)
        spl_dic[i] = shortest_path
    print(spl_dic)
    return source_nodes, spl_dic
    # for u in idx:
    #     shortest_path_length[u] = {}
    #     for v in idx:
    #         if u != v:
    #             shortest_path_length[u][v] = nx.shortest_path_length(graph, u, v)
    # end = time.time()
    # print(str(end - mid))
    # np.save("/home8t/bzx/padata/reddit/path.npy", shortest_path_length)


def linear_msbfs_train_partiton(
    csrtopo: quiver.CSRTopo, train_idx: np.ndarray, partition_num: int
):
    """基于多源BFS遍历的线性训练集划分算法
    非并行版本的线性-多源BFS训练集划分算法.与msbfs的区别是该版本不在遍历过程中完成训练集的划分,
    而是遍历完成后再遍历训练集节点，根据BFS的层数进行划分,
    并且限制了每个分区的训练节点数最多为分区平均节点数,
    贪心算法,可以得到近似解,每个分区的不规则度可能不随机

    Args:
        csrtopo (quiver.CSRTopo): _description_
        train_idx (np.ndarray): _description_
        partition_num (int): _description_
    Returns:
        dic: 训练集路径分区字典
    """
    # `{1:[3,4,5],2:[6,7,8],3:[,9,10,11],4:[12,13,14]}`
    pa_list_dic = {i: [] for i in range(partition_num)}
    train_idx_num = train_idx.shape[0]
    # 训练节点的标记,用来快速判断一个节点是否是训练节点
    train_mask: np.ndarray = np.zeros(csrtopo.node_count, dtype=bool)
    train_mask[train_idx] = True
    # 标记节点是否被访问过
    train_visited: np.ndarray = np.zeros(train_idx_num, dtype=bool)
    full_visited: np.ndarray = np.zeros(csrtopo.node_count, dtype=bool)
    # TODO根据训练节点索引找id，可以用索引和元素互换的方式存储
    train_idx = train_idx.numpy()
    train_index_dic = {id: index for index, id in enumerate(train_idx)}
    # 种子节点
    # ?时刻注意tensor和numpy的数据类型，此处统一用numpy
    # [23,57,56,43]
    start_idx = random.sample(range(train_idx_num), partition_num)
    start_idx = train_idx[start_idx]
    print(type(start_idx))
    # 根据种子节点id找分区列表的ID
    seed_index_dic = {seed: index for index, seed in enumerate(start_idx)}
    # 创建分区的当前和下一轮队列
    bfs_queues = {
        i: {"now_que": collections.deque(), "next_que": collections.deque()}
        for i in range(partition_num)
    }
    # 创建距离字典 {源节点:{目标节点:距离}}
    distance_dic = {i: {} for i in start_idx}
    # 初始化bfs队列
    for i, id in enumerate(start_idx):
        bfs_queues[i]["now_que"].append(id)
    # 创建邻居节点set，用来保存每层遍历中节点的所有邻居节点（并去重）
    neighbors_to_next = set()
    # 共遍历wordl size次,每次遍历都访问所有的训练节点后停止，获取距离字典，种子节点到每个训练节点的距离
    # 分层遍历，获取距离字典
    # 保证加入next que的节点都是未被访问过的
    # 每次从now que弹出节点后就设置visited标记
    # 如果一个节点已经被访问过，那说明其邻居节点已经被加入过next que
    # 访问到一个新节点后，将其没有被访问过的邻居加入next que中
    # 这样可以保证每个新弹出的节点都是未被访问过的，也就不需要再判断并continue
    for index, seed_id in enumerate(start_idx):
        layer_num = 0
        train_visited_count = 0
        # ?此处应该考虑使用全图节点访问次数来作为遍历结束的条件？也不见得，这样的结果也是正确的
        while train_visited_count < train_idx_num:
            while bfs_queues[index]["now_que"]:
                # 访问新节点并设置标记
                # 只有训练节点才需要加入距离字典和设置训练标记
                nid = bfs_queues[index]["now_que"].popleft()
                full_visited[nid] = True
                if train_mask[nid]:
                    train_visited[train_index_dic[nid]] = True
                    train_visited_count += 1
                    distance_dic[seed_id][nid] = layer_num
                neighbors_of_nid = csrtopo.indices[
                    csrtopo.indptr[nid] : csrtopo.indptr[nid + 1]
                ].numpy()
                neighbors_to_next.update(neighbors_of_nid)
            ...
            layer_num += 1
            # 剔除已访问的节点
            temp_set = {
                neighbor
                for neighbor in neighbors_to_next
                if full_visited[neighbor] == False
            }
            neighbors_to_next.clear()
            neighbors_to_next.update(temp_set)
            # 将未访问的节点加入next que
            # 此处甚至都不需要next que，因为已经使用了set作为下一层遍历对象的容器
            bfs_queues[index]["next_que"].extend(neighbors_to_next)
            # 交换队列
            bfs_queues[index]["now_que"], bfs_queues[index]["next_que"] = (
                bfs_queues[index]["next_que"],
                bfs_queues[index]["now_que"],
            )
            print("一层遍历结束")
        print("一次遍历结束")
        train_visited: np.ndarray = np.zeros(train_idx_num, dtype=bool)
        full_visited: np.ndarray = np.zeros(csrtopo.node_count, dtype=bool)
        neighbors_to_next.clear()
        # 输出距离字典
        with open(
            "profiler/data/debug/linear_msbfs_train_partiton_distance.txt", "w"
        ) as f:
            for key, value in distance_dic.items():
                output = "key:{},value:{}".format(key, value)
                f.write(output)
    # 以上已经获取到多源BFS得到的距离字典
    # 一下开始进行线性分区
    # 遍历训练集，获取训练集分区
    # TODO 遍历训练集时是否要打乱训练集顺序呢？
    # TODO 将邻居节点的度考虑在内
    # TODO 要考虑每个分区的节点数是否为平均数向下取整或向上取整
    # 向下取整，当填充至最后一个分区时，无需判断就可以将所有的训练节点加入最后一个分区
    ave_pa_size = train_idx_num // partition_num
    partition_nodes_count = [0] * partition_num
    copy_distance_dic = distance_dic.copy()
    for train_id in train_idx:
        # 10000是预先设定的一个很大的距离，是查找失败时返回的默认值
        seed = min(
            copy_distance_dic, key=lambda x: copy_distance_dic[x].get(train_id, 10000)
        )
        # TODO 此处需要精心地设计，如果最小值有多个，min函数只会返回第一个
        # TODO 而事实上，这种情况下需要其他的指标来决定将训练节点划分到哪个分区
        ...  # 这里可能需要添加关于度的指标

        # 获取种子节点对应的分区id
        pa_list_id = seed_index_dic[seed]
        pa_list_dic[pa_list_id].append(train_id)
        partition_nodes_count[pa_list_id] += 1
        # 复制一份distance dic，如果一个分区分配满了，就将其从距离字典中剔除
        # 距离字典中的元素大于1时，删除距离字典的元素
        if partition_nodes_count[pa_list_id] >= ave_pa_size:
            if len(copy_distance_dic) > 1:
                del copy_distance_dic[seed]
    return pa_list_dic


def linear_msbfs_train_partiton_test():
    ...
    csrtopo = quiver.CSRTopo(edge_index=edge_index)
    start = time.time()
    res = linear_msbfs_train_partiton(
        csrtopo=csrtopo, train_idx=train_idx, partition_num=4
    )
    path = "profiler/data/debug/linear_msbfs_train_partiton_res.txt"
    with open(path, "w") as f:
        for key, value in res.items():
            output = "key:{},value:{}".format(key, value)
            f.write(output)

    end = time.time()
    print(end - start)
    return res


def msbfs_train_partition(
    csrtopo: quiver.CSRTopo,
    train_idx: np.ndarray,
    partition_num: int,
):
    """非并行版本的多源BFS训练集划分算法.随机生成种子节点，每个种子节点按层依次广度优先遍历.
    使用collections中的deque,由于多源按层依次遍历,再加上遍历过程中会遇到超高度节点,会产生领域爆炸,
    进而导致负载不均衡

    Args:
        csrtopo (quiver.CSRTopo): _description_
        train_idx (np.ndarray): _description_
        partition_num (int): _description_

    Returns:
        _type_: _description_
    """
    # TODO 设置随机数种子
    ...
    ### 注意csrtopo中的为tensor，下面代码中需要的是numpy array！！！
    # 创建与GPU数量相同的列表来保存为每个GPU分配的训练节点ID
    pa_list_dic = {i: [] for i in range(partition_num)}

    train_idx_num = train_idx.shape[0]
    # 全图节点中标记为True的节点时训练节点
    train_mask: np.ndarray = np.zeros(csrtopo.node_count, dtype=bool)
    train_mask[train_idx] = True
    # accessed mask,用来标记训练节点是否被访问
    train_visited: np.ndarray = np.zeros(train_idx_num, dtype=bool)
    full_visited: np.ndarray = np.zeros(csrtopo.node_count, dtype=bool)

    train_visited_count = 0
    train_index_dic = {id: index for index, id in enumerate(train_idx)}
    # TODO 从训练集中（或者全图？）随机选取partition_num个起始节点,并加入队列作为初始值
    start_idx = random.sample(range(train_idx.shape[0]), partition_num)
    print("seed index:", start_idx)
    start_idx = train_idx[start_idx]
    print("seed train idx:", start_idx)

    # 创建各个分区的当前和下一轮的队列 (或者open-closed表？)
    bfs_queues = {
        i: {"now_que": collections.deque(), "next_que": collections.deque()}
        for i in range(partition_num)
    }
    # now_que中将种子节点入队完成初始化
    for i, id in enumerate(start_idx):
        print("seed id type:", type(id))
        bfs_queues[i]["now_que"].append(id)
        pa_list_dic[i].append(id)  # 初始化分区列表，将种子节点加入其中
    # 创建邻居节点set，用来保存每层遍历中节点的所有邻居节点（并去重）
    neighbors_to_next = set()
    # 对所有分区分层地进行循环分配，直到所有的训练节点都被分配完毕
    # 最终遍历多少层，由该while循环决定
    layer_num = 0
    start = time.time()
    while train_visited_count < train_idx_num:
        print("layer num:", layer_num)
        # 对pa_list中子list按照元素个数进行升序排序
        # sort(pa_list,sub_li_ele_num)
        # 得到排序后的列表 例如[4,2,1,3]
        # 根据为每个GPU分配的训练节点数量对pa_list进行升序排序并返回训练集分区id
        # 这样做使训练节点最少的分区能优先被分配新的训练节点
        pa_list_index_orderd = sorted(pa_list_dic, key=lambda x: len(pa_list_dic[x]))
        for pa_id in pa_list_index_orderd:
            # 维护一个数据结构，用来存放下面的一层循环中访问节点的邻居节点（可以有重复或者直接去重）
            # 使用set是乱序的，无重复的，要注意这里的顺序问题
            # TODO 注意这里的set创建和清空的方式，是在每次循环开始前创建一个新的set，还是在前面创建set，在这里清空set
            neighbors_to_next.clear()
            # 首先遍历now queue中所有节点，每次迭代时判断是否为train，并设置visited信息
            # 然后将当前节点的邻居添加到next queue中（是否添加随机性？）
            # 如下是一个分区的一层的遍历
            while bfs_queues[pa_id]["now_que"]:
                # print("layer:{},pa id:{}".format(layer_num, pa_id))
                nid = bfs_queues[pa_id]["now_que"].popleft()
                # 找到nid的所有邻居节点，从csr格式中获取,将tensor转为numpy
                neighbor_of_nid = (
                    csrtopo.indices[csrtopo.indptr[nid] : csrtopo.indptr[nid + 1]]
                ).numpy()
                # 无论节点是否被访问过，都加入下一层队列中，也就是多个遍历碰撞后继续重叠遍历
                # TODO 此处有bug，每层遍历时，对当前层也就是now que中的每个节点，都将其所有邻居节点放入队列，
                # 但是每个节点的邻居节点可能是存在重复的，应该在每层去重后再添加到下一层的队列
                # TODO 可以考虑使用update方法，直接将可迭代的列表添加至set
                # 此处添加下一层访问的节点时并没有考虑多个源遍历发生碰撞的情况，而是默认碰撞后继续重叠遍历
                # 可以考虑为碰撞的节点添加随机性，也要考虑添加了随机性后算法是否收敛的问题
                # TODO重新分配发生碰撞的训练节点，同时考虑邻居的度和分区已有训练节点数
                # for neighbor in neighbor_of_nid:
                # bfs_queues[pa_id]["next_que"].put(neighbor)  # 此处有问题，大量重复节点
                # neighbors_to_next.add(neighbor)
                # set.update使用c语言编写，比起for循环效率要高
                neighbors_to_next.update(neighbor_of_nid)
                # 处理当前节点的全局访问mask 这一块代码不是必须的
                if full_visited[nid] == True:
                    continue
                else:
                    full_visited[nid] = True
                # 处理当前节点的训练集访问mask
                if train_mask[nid]:  # TODO
                    if train_visited[train_index_dic[nid]] == True:
                        continue
                    # 将训练节点加入到分区列表中 最终结果
                    # TODO 在最后一层遍历时，可能只完成部分分区的遍历就划分了所有的训练节点，在现有算法的条件下
                    # 最后一层中未循环的分区节点数较少，分区间负载不均衡
                    # 一个分区中添加一个或若干个训练节点时，就应该暂停该分区当前层的遍历，开始进行下一个分区
                    # 这样就保证了训练节点少的分区也有机会分到节点
                    # 本质上来说，是通过对层遍历的细粒度划分来实现分区间训练节点的负载均衡
                    ...
                    pa_list_dic[pa_id].append(nid)
                    train_visited[train_index_dic[nid]] = True
                    train_visited_count += 1
                    # 该分区得到一个新的训练节点时就中断遍历，开始为下一个分区搜索
                    # break
            # 将上面的一层循环中访问节点的所有邻居节点去重后加入到next que中
            for neighbor in neighbors_to_next:
                bfs_queues[pa_id]["next_que"].append(neighbor)
            # 以上一层遍历结束，下面更新now队列和next队列的内容
            # TODO 添加判断条件 上一层遍历结束的flag是该分区的now que为空，只有now que为空时才更新下一层队列到当前队列
            # 下面的两个语句也是为了实现细粒度的划分算法，可以保存当前分区当前层的遍历状态
            # if bfs_queues[pa_id]["now_que"]:
            #     continue
            bfs_queues[pa_id]["now_que"], bfs_queues[pa_id]["next_que"] = (
                bfs_queues[pa_id]["next_que"],
                bfs_queues[pa_id]["now_que"],
            )
        layer_num += 1
        for i in range(partition_num):
            print("partition {},node num:{}".format(i, len(pa_list_dic[i])))

    end = time.time()
    print("time:", int(end - start), "s")
    return pa_list_dic


def msbfs_train_partition_test():
    csrtopo = quiver.CSRTopo(edge_index=edge_index)
    res = msbfs_train_partition(csrtopo=csrtopo, train_idx=idx, partition_num=4)
    for key, value in res.items():
        print("pa id:{},train node num:{}".format(key, len(value)))
    np.save("/home8t/bzx/padata/reddit/pa_li_dic" + str(time.time()) + ".npy", res)


def spl_train_partition_demo():
    start = time.time()
    G = nx.fast_gnp_random_graph(1000000, 0.001)
    end = time.time()
    print(end - start)
    # plt.figure(figsize=(50, 50))
    # nx.draw_networkx(G, font_size=2)
    # plt.tight_layout()
    # plt.savefig(
    #     "a.png",
    #     dpi=300,
    # )

    source_nodes = random.sample(G.nodes(), 4)
    print(type(source_nodes))
    spl_dic = {}
    start = time.time()
    for i, start_node in enumerate(source_nodes):
        shortest_path = nx.single_source_dijkstra_path_length(G, start_node)
        print(type(shortest_path))
        spl_dic[i] = shortest_path
    end = time.time()
    print(end - start)
    # print(spl_dic)


if __name__ == "__main__":
    set_random_seed(1998)
    # demo_test()
    # reddit_test()
    # train_idx_partition(1, train_idx=idx, partition_num=4)
    # msbfs_train_partition_test()
    # spl_train_partition_demo()
    linear_msbfs_train_partiton_test()
