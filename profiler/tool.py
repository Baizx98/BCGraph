import os
import os.path as osp
from typing import List, Union

import numpy as np
import scipy.sparse as sp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import quiver


def reindex_nid_by_hot_metric(
    hot_metric: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(hot_metric, torch.Tensor):
        _, pre_order = torch.sort(hot_metric)
        return pre_order
    if isinstance(hot_metric, List):
        prev_order_list = []
        for li in hot_metric:
            _, temp = torch.sort(li, descending=True)
            prev_order_list.append(temp)
        return prev_order_list
    # logging.info("reindex done")


def get_dataset_save_path(dataset_name: str = "") -> str:
    return osp.join("/home8t/bzx", "data", dataset_name)


def get_profiler_data_save_path(
    file_name: str, item="", profiler_data_path="profiler/data/"
) -> str:
    """获取分析结果的保存路径
    Args:
        file_name (str): 文件名
    Returns:
        str: 分析结果的保存路径
    """
    if not osp.exists(profiler_data_path + item):
        os.mkdir(profiler_data_path + item)
    profiler_data_path = osp.join(profiler_data_path, item, file_name)
    return profiler_data_path


def ogbn2pagraph(i: int):
    dataset_name = ["ogbn-products", "ogbn-papers100M"]
    dataset_path = "/home8t/bzx/data/"
    padataset_path = "/home8t/bzx/padata"
    dataset = PygNodePropPredDataset(dataset_name[i], dataset_path)
    data = dataset[0]
    node_num = data.x.shape[0]
    print("vnum:", node_num)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].numpy()
    valid_idx = split_idx["valid"].numpy()
    test_idx = split_idx["test"].numpy()
    train_mask = np.full(node_num, False, dtype=bool)
    train_mask[train_idx] = True
    valid_mask = np.full(node_num, False, dtype=bool)
    valid_mask[valid_idx] = True
    test_mask = np.full(node_num, False, dtype=bool)
    valid_mask[valid_idx] = True
    labels: torch.Tensor = data.y.squeeze().numpy()
    print(labels)
    x = data.x
    print(x.shape)
    edge_index = data.edge_index
    print("begin to save")
    np.save(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "train.npy"),
        train_mask,
    )
    np.save(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "test.npy"),
        test_mask,
    )
    np.save(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "val.npy"),
        valid_mask,
    )
    np.save(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "labels.npy"),
        labels,
    )
    np.save(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "feat.npy"), x
    )
    adj = sp.coo_matrix(
        (torch.ones(edge_index.shape[1]), edge_index),
        shape=(data.num_nodes, data.num_nodes),
    )
    print(type(adj))
    # adj = adj.tocsr()
    sp.save_npz(
        os.path.join(padataset_path, dataset_name[i].replace("-", "_"), "adj.npz"),
        adj,
    )


def livejournal2edgeindex():
    with open(
        "/home8t/bzx/data/livejournal/soc-LiveJournal1.txt", "r", encoding="utf-8"
    ) as f:
        lines = f.readlines()
    lines = lines[4:]
    sources = []
    targets = []
    for line in lines:
        # a = line.strip().split("\t")
        # print(a)
        source, target = line.strip().split("\t")
        sources.append(int(source))
        targets.append(int(target))
    tensor = np.vstack([sources, targets])
    edge_index = torch.from_numpy(tensor).int()
    torch.save(
        edge_index, os.path.join("/home8t/bzx/data", "livejournal", "edge_index.pt")
    )


def split_dataset(vnum, path):
    nids = np.arange(vnum)
    np.random.shuffle(nids)
    train_len = int(vnum * 0.65)
    val_len = int(vnum * 0.1)
    test_len = vnum - train_len - val_len
    # train mask
    train_mask = np.zeros(vnum, dtype=int)
    train_mask[nids[0:train_len]] = 1
    # val mask
    val_mask = np.zeros(vnum, dtype=int)
    val_mask[nids[train_len : train_len + val_len]] = 1
    # test mask
    test_mask = np.zeros(vnum, dtype=int)
    test_mask[nids[-test_len:]] = 1
    # save
    if path is not None:
        np.save(os.path.join(path, "train.npy"), train_mask)
        np.save(os.path.join(path, "val.npy"), val_mask)
        np.save(os.path.join(path, "test.npy"), test_mask)
    return train_mask, val_mask, test_mask


def random_feature(vnum, feat_size, path):
    feat_mat = np.random.random((vnum, feat_size)).astype(np.float32)
    np.save(path, feat_mat)


def random_label(vnum, class_num, path):
    labels = np.random.randint(class_num, size=vnum)
    np.save(path, labels)


if __name__ == "__main__":
    path = "/home8t/bzx/data/livejournal"
    edge_index = torch.load("/home8t/bzx/data/livejournal/edge_index.pt")
    csr_topo = quiver.CSRTopo(edge_index=edge_index)
    vnum = csr_topo.node_count
    a, b, c = split_dataset(vnum, path)
    print(a)
