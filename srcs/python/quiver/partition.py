import torch
import shutil
import os
from typing import List
import quiver.utils as quiver_util


__all__ = ["quiver_partition_feature", "load_quiver_feature_partition"]


QUIVER_MAGIC_NUMBER = 256


def partition_feature_without_replication(probs: List[torch.Tensor], chunk_size: int):
    """Partition node with node access distribution. 
    The result will cause no replication between each parititon.

    这段代码实现了基于节点访问分布（node access distribution）的无重复特征分割方法，其输入参数为一个概率分布列表和一个chunk size，输出为一个ID列表，其中每个ID列表代表一个特征分割（feature partition）的结果，以及分割后每个节点的概率分布。具体流程如下：

    将概率分布列表中的每个张量转移到当前设备（这里假定使用GPU）。
    计算chunk_num，即将所有节点分成的块数。对于每个块，其包含从当前块的起始位置开始的 chunk_size 个节点。每个块对应一个包含节点ID的张量 chunk。
    对于每个块，计算每个分区（partition）的节点访问概率之和 probs_sum_chunk。对于每个分区，假设当前分区为 src_rank，对于除了当前分区以外的所有分区，将其概率分布相加，然后将结果减去当前分区的概率分布。最后将得到一个包含当前块中所有节点的 probs_sum_chunk 张量列表。这里通过将除当前分区以外的所有分区的概率分布相加再减去当前分区的概率分布的方式，保证了分割后每个节点只被分配到一个分区中。
    对于每个分区，找到其应该选择的节点，即在当前块中选择其概率分布最高的节点。将这些节点的ID存储在一个张量 pick_ids 中，并将其添加到结果列表 res 中。同时，将这些节点在所有分区的 probs_sum_chunk 张量列表中的值设置为 -1，以确保下一次选择不会再次选择到这些节点。
    重复步骤4直到所有节点都被分配到一个分区为止。
    将每个分区的ID列表拼接起来，返回结果。

    Args:
        probs (torch.Tensor): node access distribution
        chunk_size (int): chunk_size 

    Returns:
        [torch.Tensor]: list of IDs for each partition

    """

    device = torch.cuda.current_device()
    partitioned_num = len(probs)

    probs = [prob.to(device) for prob in probs]
    total_node_num = probs[0].size(0)

    res = [[] for _ in range(partitioned_num)]

    blob_size = chunk_size * partitioned_num
    chunk_num = (total_node_num + chunk_size - 1) // chunk_size

    current_chunk_start_pos = 0
    current_partition_idx = 0
    for _ in range(chunk_num):
        current_chunk_end_pos = min(
            total_node_num, current_chunk_start_pos + blob_size)
        current_chunk_size = current_chunk_end_pos - current_chunk_start_pos
        chunk = torch.arange(current_chunk_start_pos,
                             current_chunk_end_pos, device=device)
        probs_sum_chunk = [
            torch.zeros(current_chunk_size, device=device) + 1e-6 for _ in range(partitioned_num)
        ]
        for src_rank in range(partitioned_num):
            for dst_rank in range(partitioned_num):
                if dst_rank == src_rank:
                    probs_sum_chunk[src_rank] += probs[dst_rank][chunk] * \
                        partitioned_num
                else:
                    probs_sum_chunk[src_rank] -= probs[dst_rank][chunk]
        assigned_node_size = 0
        per_partition_size = chunk_size
        for partition_idx in range(current_partition_idx, current_partition_idx + partitioned_num):
            partition_idx = partition_idx % partitioned_num
            actual_per_partition_size = min(
                per_partition_size, current_chunk_size - assigned_node_size)
            _, sorted_res_order = torch.sort(
                probs_sum_chunk[partition_idx], descending=True)
            pick_chunk_part = sorted_res_order[:actual_per_partition_size]
            pick_ids = chunk[pick_chunk_part]
            res[partition_idx].append(pick_ids)
            for idx in range(partitioned_num):
                probs_sum_chunk[idx][pick_chunk_part] = -1
            assigned_node_size += actual_per_partition_size
        current_partition_idx += 1
        current_chunk_start_pos += current_chunk_size

    for partition_idx in range(partitioned_num):
        res[partition_idx] = torch.cat(res[partition_idx])
    return res, probs


def quiver_partition_feature(probs: torch.Tensor, result_path: str, cache_memory_budget=0, per_feature_size=0, chunk_size=QUIVER_MAGIC_NUMBER):
    """
    它基于访问概率对图形特征张量进行分区，并生成一个结果文件夹，其中包含每个分区的分区特征张量和缓存特征张量。

    以下是该函数的详细说明：

    输入参数为：

    probs：图形中每个节点的访问概率张量。
    result_path：分区特征张量和缓存特征张量将保存的路径。
    cache_memory_budget：用户指定的缓存热点特征的内存预算。
    per_feature_size：用户特征的每个特征大小。
    chunk_size：一个常量值，用于确定分区特征张量的每个块的大小。
    该函数首先检查结果文件夹是否已存在。如果存在，则会提示用户确认是否删除文件夹并继续或退出该函数。

    分区数等于probs张量的长度。

    为结果文件夹中的每个分区创建一个文件夹。

    该函数根据用户指定的内存预算和每个特征大小计算可以缓存的特征数量。

    然后，该函数使用partition_feature_without_replication函数对特征张量进行无重复分区。

    如果启用了缓存，则根据访问概率计算要缓存哪些特征，并将它们缓存。

    将每个分区的分区特征张量和缓存特征张量保存在它们各自的文件夹中。

    该函数返回分区图书，它是一个张量，指示每个节点属于哪个分区，以及每个分区的分区特征张量和缓存特征张量。

    Partition graph feature based on access probability and generate result folder. The final result folder will be like:

    -result_path
        -partition_0
            -partition_res.pth
            -cache_res.pth
        -partition_1
            -partition_res.pth
            -cache_res.pth
        -partition_2
            -partition_res.pth
            -cache_res.pth
        ...

    Args:
        probs:
        result_path (str): path for partition result
        cache_memory_budget (Union[str, int, float]): user-specified memory budget for caching hot feature
        per_feature_size (Union[str, int, float]): per-feature size for user's feature

    Returns:
        partition_book (torch.Tensor): Indicates which partition_idx a node belongs to
        feature_partition_res (torch.Tensor): partitioned feature result
        feature_cache_res (torch.Tensor): cached feature result
    """

    if os.path.exists(result_path):
        res = input(
            f"{result_path} already exists, enter Y/N to continue, If continue, {result_path} will be deleted:")
        res = res.upper()
        if res == "Y":
            shutil.rmtree(result_path)
        else:
            print("exiting ...")
            exit()

    partition_num = len(probs)

    # create result folder
    for partition_idx in range(partition_num):
        os.makedirs(os.path.join(
            result_path, f"feature_partition_{partition_idx}"))

    # calculate cached feature count
    cache_memory_budget_bytes = quiver_util.parse_size(cache_memory_budget)
    per_feature_size_bytes = quiver_util.parse_size(per_feature_size)
    cache_count = int(cache_memory_budget_bytes /
                      (per_feature_size_bytes + 1e-6))
    per_partition_cache_count = cache_count // partition_num

    partition_book = torch.zeros(
        probs[0].shape, dtype=torch.int64, device=torch.cuda.current_device())
    partition_res, changed_probs = partition_feature_without_replication(
        probs, chunk_size)

    cache_res = [None] * partition_num

    if cache_count > 0:
        for partition_idx in range(partition_num):
            _, prev_order = torch.sort(
                changed_probs[partition_idx], descending=True)
            cache_res[partition_idx] = prev_order[: per_partition_cache_count]

    for partition_idx in range(partition_num):
        partition_result_path = os.path.join(
            result_path, f"feature_partition_{partition_idx}", "partition_res.pth")
        cache_result_path = os.path.join(
            result_path, f"feature_partition_{partition_idx}", "cache_res.pth")
        partition_book[partition_res[partition_idx]] = partition_idx
        torch.save(partition_res[partition_idx], partition_result_path)
        torch.save(cache_res[partition_idx], cache_result_path)

    partition_book_path = os.path.join(
        result_path, f"feature_partition_book.pth")
    torch.save(partition_book, partition_book_path)

    return partition_book, partition_res, cache_res


def load_quiver_feature_partition(partition_idx: int, result_path: str):
    """
    Load partition result for partition ${partition_idx}

    它用于从保存的文件中加载分区结果。

    函数的输入参数包括：

    partition_idx：分区的索引值
    result_path：保存分区结果的路径
    函数返回三个张量，分别是：

    partition_book：指示每个节点属于哪个分区的张量
    partition_res：属于这个分区的节点的索引张量
    cache_res：属于这个分区且已被缓存的节点的索引张量
    在函数内部，首先会检查result_path路径是否存在，如果不存在则会抛出异常。然后会根据partition_idx和result_path构造分区结果文件和缓存结果文件的路径。接着，函数会使用torch.load函数加载分区结果、缓存结果和分区标记的张量，然后返回这些张量。

    Args:
        partition_idx (int): Partition idx
        partition_result_path (str): partition result path

    Returns:
        partition_book (torch.Tensor): partition_book indicates which partition_idx a node belongs to
        partition_res (torch.Tensor): node indexes belong to this partition
        cache_res (torch.Tensor): cached node indexes belong to this partition

    """

    if not os.path.exists(result_path):
        raise Exception("Result path not exists")

    partition_result_path = os.path.join(
        result_path, f"feature_partition_{partition_idx}", "partition_res.pth")
    cache_result_path = os.path.join(
        result_path, f"feature_partition_{partition_idx}", "cache_res.pth")
    partition_book_path = os.path.join(
        result_path, f"feature_partition_book.pth")

    partition_book = torch.load(partition_book_path)
    partition_res = torch.load(partition_result_path)
    cache_res = torch.load(cache_result_path)

    return partition_book, partition_res, cache_res
