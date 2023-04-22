import os
import os.path as osp
from typing import List, Union

import torch


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


if __name__ == "__main__":
    print(get_dataset_save_path())
