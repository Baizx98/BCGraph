# 一阶因子参数
dataset = "ogbn-products"
node_num = 2400000  # 图数据集的节点数量
pagraph_node_num = 1000000  # pagraph 分区一个子图中节点的数量
feat_ele_size = 4  # 特征向量中一个元素的大小 单位为byte
feat_dim = 6000  # 特征向量的维度

world_size = 4  # GPU的个数
cache_capacity = 1  # 单个GPU的缓存空间 单位为GB

local_access_latency = 0  # 访问本地GPU内存的开销
remote_access_latency = 1  # 访问远程GPU内存的开销
cpu_access_latency = 10  # 访问CPU内存的开销

mini_batch_size = 1024  # 一批训练节点的数量
sample_size = [25, 10]  # 采样的深度和数量

# 可变系数
remote_access_latency_alpha = 1  # 访问远程GPU内存的开销的可扩展系数，与world_size有关，GPU越多，远程访问开销会逐渐增加
cpu_contention_alpha = 1  # CPU的争用系数，GPU越多，对CPU资源的竞争越大，总的延迟也就越高,与world_size有关
partition_redundancy_alpha = 1  # pagraph的分区算法中，分区后所有节点之和与数据集节点数之比，与数据集和分区数量（也就是GPU数量）有关
cache_repetition_alpha = 1  # GPU间缓存重复节点的系数


feat_size = feat_ele_size * feat_dim  # 一个特征的大小
cache_num = int(cache_capacity * 1024 * 1024 * 1024 / feat_size)  # 单个GPU可缓存的节点特征数量
pagraph_cache_ratio = cache_num / pagraph_node_num  # 单个GPU上缓存节点占（子）图中节点的比例
nvlink_cache_ratio = cache_num * 4 / node_num
print("pagraph cache ratio:%-5f" % pagraph_cache_ratio)
print("nvlink cache ratio:%-5f" % nvlink_cache_ratio)
# access_node_num = 10000   每个batch单个GPU上访问的采样子图的节点数，与mini_batch_size和sample_size有关,数据集

partition_time = 0  # 分区算法的时间开销，pagraph的分区算法与数据集的规模node_num和分区数量world_size有关


def pagraph_latency():
    # PaGraph的分区 假设不同子图的训练是负载均衡的 这样的好处体现在子图内的局部性更好，命中率更高
    # 分区的节点存在冗余
    partition_hit_ratio = get_hit_ratio(
        "pagraph", dataset=dataset, cache_ratio=pagraph_cache_ratio
    )
    print("pagraph hit ratio", partition_hit_ratio)
    access_node_num = get_access_node_num("pagraph", dataset=dataset, batch_size=1024)
    print("pagraph access node num:", access_node_num)
    return (
        partition_time
        + access_node_num * partition_hit_ratio * local_access_latency
        + (access_node_num - access_node_num * partition_hit_ratio)
        * cpu_access_latency
        * cpu_contention_alpha
    )


def nvlink_latency():
    local_hit_ratio = get_hit_ratio(
        "nvlink", dataset=dataset, cache_ratio=nvlink_cache_ratio
    )
    print("nvlink hit ratio", local_hit_ratio)
    # native或random分区 统一缓存空间
    # 其中本地命中率和单个远程GPU命中率是相等的，也就是remote_hit_ratio=local_hit_ratio*(world-1)
    remote_hit_ratio = min(0.75, (world_size - 1) * local_hit_ratio)
    access_node_num = get_access_node_num("nvlink", dataset=dataset, batch_size=1024)
    print("nvlink access node num:", access_node_num)

    # access_node_num = 189214
    return (
        partition_time
        + access_node_num * local_hit_ratio * local_access_latency
        + access_node_num
        * remote_hit_ratio
        * remote_access_latency
        * remote_access_latency_alpha
        + access_node_num
        * (1 - local_hit_ratio - remote_hit_ratio)
        * cpu_access_latency
        * cpu_contention_alpha
    )


def get_hit_ratio(scheme, dataset, cache_ratio):
    x = cache_ratio
    hit_dict = {
        ("pagraph", "reddit", 4): 1.2036 * x**4
        - 1.1528 * x**3
        - 2.0944 * x**2
        + 3.0699 * x
        - 0.0234,
        ("pagraph", "ogbn-products", 4): -3.3902 * x**4
        + 8.9584 * x**3
        - 8.5205 * x**2
        + 3.4675 * x
        + 0.4814,
        ("nvlink", "reddit", 4): -0.2469 * x**2 + 0.4938 * x + 0.0033,
        ("nvlink", "ogbn-products", 4): 0.2567 * x**3
        - 0.7092 * x**2
        + 0.6748 * x
        + 0.0285,
    }
    return hit_dict.get((scheme, dataset, world_size), 0)


def get_partition_time(dataset: str, partition_num: int):
    time = 0
    if dataset == "Reddit":
        if partition_num == 2:
            time = 2
        if partition_num == 4:
            time = 4
        if partition_num == 8:
            time = 8
    if dataset == "ogbn-products":
        if partition_num == 2:
            time = 2
        if partition_num == 4:
            time = 4
        if partition_num == 8:
            time = 8
    if dataset == "ogbn-papers100M":
        if partition_num == 2:
            time = 2
        if partition_num == 4:
            time = 4
        if partition_num == 8:
            time = 8
    return 0


def get_access_node_num(scheme, dataset, batch_size):
    hit_dict = {
        ("pagraph", "reddit", 4, 1024): 51612,
        ("pagraph", "ogbn-products", 4, 1024): 37049,
        ("nvlink", "reddit", 4, 1024): 107234,
        ("nvlink", "ogbn-products", 4, 1024): 189117,
    }
    return hit_dict.get((scheme, dataset, world_size, batch_size), 0)


a = nvlink_latency()
print("nvlink latency", int(a))
b = pagraph_latency()
print("pagraph latency", int(b))
