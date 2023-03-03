import torch
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo
from quiver.utils import reindex_feature, CSRTopo, parse_size
from typing import List
import numpy as np
from torch._C import device

__all__ = ["Feature", "DistFeature", "PartitionInfo", "DeviceConfig"]


class DeviceConfig:
    def __init__(self, gpu_parts, cpu_part):
        self.gpu_parts = gpu_parts
        self.cpu_part = cpu_part


class Feature(object):
    """Feature partitions data onto different GPUs' memory and CPU memory and does feature collection with high performance.
    You will need to set `device_cache_size` to tell Feature how much data it can cached on GPUs memory. By default, it will partition data by your  `device_cache_size`, if you want to cache hot data, you can pass
    graph topology `csr_topo` so that Feature will reorder all data by nodes' degree which we expect to provide higher cache hit rate and will offer better performance with regard to cache random data.

    ```python
    >>> cpu_tensor = torch.load("cpu_tensor.pt")
    >>> feature = Feature(0, device_list=[0, 1], device_cache_size='200M')
    >>> feature.from_cpu_tensor(cpu_tensor)
    >>> choose_idx = torch.randint(0, feature.size(0), 100)
    >>> selected_feature = feature[choose_idx]
    ```
    Args:
        rank (int): device for feature collection kernel to launch
        device_list ([int]): device list for data placement
        device_cache_size (Union[int, str]): cache data size for each device, can be like `0.9M` or `3GB`
        cache_policy (str, optional): cache_policy for hot data, can be `device_replicate` or `p2p_clique_replicate`, choose `p2p_clique_replicate` when you have NVLinks between GPUs, else choose `device_replicate`. (default: `device_replicate`)
        csr_topo (quiver.CSRTopo): CSRTopo of the graph for feature reordering

    """

    def __init__(self,
                 rank: int,
                 device_list: List[int],
                 device_cache_size: int = 0,
                 cache_policy: str = 'device_replicate',
                 csr_topo: CSRTopo = None):
        assert cache_policy in [
            "device_replicate", "p2p_clique_replicate"
        ], f"Feature cache_policy should be one of [device_replicate, p2p_clique_replicate]"
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.clique_tensor_list = {}
        self.rank = rank
        self.topo = Topo(self.device_list)
        self.csr_topo = csr_topo
        self.feature_order = None
        self.ipc_handle_ = None
        self.mmap_handle_ = None
        self.disk_map = None
        assert self.clique_device_symmetry_check(
        ), f"\n{self.topo.info()}\nDifferent p2p clique size NOT equal"

    def clique_device_symmetry_check(self):
        if self.cache_policy == "device_replicate":
            return True
        print(
            "WARNING: You are using p2p_clique_replicate mode, MAKE SURE you have called quiver.init_p2p() to enable p2p access"
        )
        if len(self.topo.p2pClique2Device.get(1, [])) == 0:
            return True
        if len(self.topo.p2pClique2Device.get(1, [])) == len(
                self.topo.p2pClique2Device[0]):
            return True
        return False

    def cal_size(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        """根据GPU Cache的总空间计算可缓存的元素数量

        Args:
            cpu_tensor (torch.Tensor): tensor
            cache_memory_budget (int): GPU显存中可以用于缓存数据的空间大小

        Returns:
            int: 可缓存的tensor数量
        """
        element_size = cpu_tensor.shape[1] * cpu_tensor.element_size()
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        """该函数是Feature类的方法，用于将传入的cpu_tensor按照给定的cache_memory_budget分成两部分，并将第一部分数据缓存到GPU上。具体实现是根据cache_memory_budget计算出可以缓存到GPU上的元素个数cache_size，然后通过切片将cpu_tensor分为两部分，前cache_size个元素存储到GPU上，后面的元素则存储在CPU内存中，并将这两个部分作为一个列表返回。

        Args:
            cpu_tensor (torch.Tensor): cpu_tensor
            cache_memory_budget (int): GPU 缓存空间大小

        Returns:
            List: 特征数据列表，其中有两个元素，第一个将缓存至GPU，第二个将储存至CPU内存中
        """

        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size:]]

    def set_mmap_file(self, path, disk_map):
        self.lazy_init_from_ipc_handle()
        self.mmap_handle_ = np.load(path, mmap_mode='r')
        self.disk_map = disk_map.to(self.rank)

    def read_mmap(self, ids):
        ids = ids.cpu().numpy()
        res = torch.from_numpy(self.mmap_handle_[ids])
        res = res.to(device=self.rank, dtype=torch.float32)
        return res

    def from_mmap(self, np_array, device_config):
        """Create quiver.Feature from a mmap numpy array and partition config

        这段代码实现了从一个 mmap Numpy 数组和分区配置中创建一个 ShardTensor 对象。主要流程如下：

        首先，判断分区配置的 GPU 分区数量是否和 device_list 的长度相等。然后，根据 cache_policy 的不同，对每个设备构建对应的分片。

        如果 cache_policy 为 "device_replicate"，则对于每个设备，如果该设备对应的分区是一个 tensor，则直接将该 tensor 转换成 float32 类型后添加到 ShardTensor 对象中；如果该设备对应的分区是一个字符串，则将该字符串指向的文件读取为 tensor，然后再添加到 ShardTensor 对象中。

        如果 cache_policy 不为 "device_replicate"，则需要先根据设备的 rank 判断该设备所在的 clique，然后对每个 clique 构建一个 ShardTensor 对象。接着，对于每个 clique 中的设备，按照和 "device_replicate" 情况下相同的方式添加分片到对应的 ShardTensor 对象中。

        最后，根据分区配置中的 CPU 分区信息，创建一个 CPU tensor，并根据 cache_policy 的不同，将该 tensor 添加到相应的 ShardTensor 对象中。

        总之，该方法的作用是将分区好的 tensor 转换成 ShardTensor 对象，用于分布式计算。

        Args:
            np_array (numpy.ndarray): mmap numpy array
            device_config (quiver.feature.DeviceConfig): device partitionconfig
        """
        assert len(device_config.gpu_parts) == len(self.device_list)
        if self.cache_policy == "device_replicate":
            for device in self.device_list:
                if isinstance(device_config.gpu_parts[device], torch.Tensor):
                    if np_array is None:
                        cache_part = device_config.gpu_parts[device].to(
                            dtype=torch.float32)
                    else:
                        cache_ids = device_config.gpu_parts[device].numpy()
                        cache_part = torch.from_numpy(
                            np_array[cache_ids]).to(dtype=torch.float32)
                elif isinstance(device_config.gpu_parts[device], str):
                    cache_part = torch.load(device_config.gpu_parts[device])
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor
                del cache_part

        elif self.cache_policy == "p2p_clique_replicate":
            clique0_device_list = self.topo.p2pClique2Device.get(0, [])
            clique1_device_list = self.topo.p2pClique2Device.get(1, [])

            if len(clique0_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique0_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                for idx, device in enumerate(clique0_device_list):
                    if isinstance(device_config.gpu_parts[device],
                                  torch.Tensor):
                        if np_array is None:
                            cache_part = device_config.gpu_parts[device].to(
                                dtype=torch.float32)
                        else:
                            cache_ids = device_config.gpu_parts[device].numpy()
                            cache_part = torch.from_numpy(
                                np_array[cache_ids]).to(dtype=torch.float32)
                    elif isinstance(device_config.gpu_parts[device], str):
                        cache_part = torch.load(
                            device_config.gpu_parts[device])
                    shard_tensor.append(cache_part, device)
                    self.device_tensor_list[device] = shard_tensor
                    del cache_part

                self.clique_tensor_list[0] = shard_tensor

            if len(clique1_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique1_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                for idx, device in enumerate(clique1_device_list):
                    if isinstance(device_config.gpu_parts[device],
                                  torch.Tensor):
                        if np_array is None:
                            cache_part = device_config.gpu_parts[device].to(
                                dtype=torch.float32)
                        else:
                            cache_ids = device_config.gpu_parts[device].numpy()
                            cache_part = torch.from_numpy(
                                np_array[cache_ids]).to(dtype=torch.float32)
                    elif isinstance(device_config.gpu_parts[device], str):
                        cache_part = torch.load(
                            device_config.gpu_parts[device])
                    shard_tensor.append(cache_part, device)
                    self.device_tensor_list[device] = shard_tensor
                    del cache_part

                self.clique_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if isinstance(device_config.cpu_part, torch.Tensor):
            if np_array is None:
                self.cpu_part = device_config.cpu_part
            else:
                cache_ids = device_config.cpu_part.numpy()
                self.cpu_part = torch.from_numpy(
                    np_array[cache_ids]).to(dtype=torch.float32)
        elif isinstance(device_config.cpu_part, str):
            self.cpu_part = torch.load(device_config.cpu_part)
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(
                    self.rank, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list.get(
                    clique_id, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.clique_tensor_list[clique_id] = shard_tensor

    def from_cpu_tensor(self, cpu_tensor: torch.Tensor):
        """Create quiver.Feature from a pytorh cpu float tensor

        函数的具体操作如下：

        根据设定的缓存策略和设备缓存大小，计算出缓存所需的内存和缓存比例。
        根据输入的CSR拓扑（如果有）对输入的CPU Tensor进行重排序（shuffle）。
        将输入的CPU Tensor分成两部分：一部分用于缓存（cache_part），一部分用于CPU计算（cpu_part）。
        如果使用的是“device_replicate”策略，将缓存数据复制到所有设备上，并存储为ShardTensor的形式。
        如果使用的是其他策略，将缓存数据按照拓扑分配到不同的设备上，并存储为ShardTensor的形式。
        将剩余的CPU Tensor数据按照拓扑分配到对应的设备上，并存储为ShardTensor的形式。

        Args:
            cpu_tensor (torch.FloatTensor): input cpu tensor
        """
        if self.cache_policy == "device_replicate":
            cache_memory_budget = parse_size(self.device_cache_size)
            shuffle_ratio = 0.0
        else:
            cache_memory_budget = parse_size(
                self.device_cache_size) * len(self.topo.p2pClique2Device[0])
            shuffle_ratio = self.cal_size(
                cpu_tensor, cache_memory_budget) / cpu_tensor.size(0)

        print(
            f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / cpu_tensor.element_size()))}% data cached"
        )
        if self.csr_topo is not None:
            if self.csr_topo.feature_order is None:
                cpu_tensor, self.csr_topo.feature_order = reindex_feature(
                    self.csr_topo, cpu_tensor, shuffle_ratio)
            self.feature_order = self.csr_topo.feature_order.to(self.rank)
        cache_part, self.cpu_part = self.partition(cpu_tensor,
                                                   cache_memory_budget)
        self.cpu_part = self.cpu_part.clone()
        if cache_part.shape[0] > 0 and self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor

        elif cache_part.shape[0] > 0:
            clique0_device_list = self.topo.p2pClique2Device.get(0, [])
            clique1_device_list = self.topo.p2pClique2Device.get(1, [])

            block_size = self.cal_size(
                cpu_tensor,
                cache_memory_budget // len(self.topo.p2pClique2Device[0]))

            if len(clique0_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique0_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique0_device_list):
                    if idx == len(clique0_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(
                            cache_part[cur_pos:cur_pos + block_size], device)
                        cur_pos += block_size

                self.clique_tensor_list[0] = shard_tensor

            if len(clique1_device_list) > 0:
                print(
                    f"LOG>>> GPU {clique1_device_list} belong to the same NUMA Domain"
                )
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for idx, device in enumerate(clique1_device_list):
                    if idx == len(clique1_device_list) - 1:
                        shard_tensor.append(cache_part[cur_pos:], device)
                    else:

                        shard_tensor.append(
                            cache_part[cur_pos:cur_pos + block_size], device)
                        cur_pos += block_size

                self.clique_tensor_list[1] = shard_tensor

        # 构建CPU Tensor
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(
                    self.rank, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list.get(
                    clique_id, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.clique_tensor_list[clique_id] = shard_tensor

    def set_local_order(self, local_order):
        """ Set local order array for quiver.Feature

        用于为Quiver.Feature对象设置本地顺序。它需要一个torch.Tensor类型的local_order参数，该参数包含特征的原始索引。在方法内部，首先使用torch.arange方法创建一个本地的索引范围，然后使用torch.zeros_like方法创建一个与本地索引范围相同的全零张量作为特征顺序张量。接下来，将local_order参数转换为Feature对象的设备，并使用它在特征顺序张量中填充相应的值。最终，Feature对象的特征顺序张量将包含local_order参数中特征的本地索引。

        Args:
            local_order (torch.Tensor): Tensor which contains the original indices of the features

        """
        local_range = torch.arange(end=local_order.size(0),
                                   dtype=torch.int64,
                                   device=self.rank)
        self.feature_order = torch.zeros_like(local_range)
        self.feature_order[local_order.to(self.rank)] = local_range

    def __getitem__(self, node_idx: torch.Tensor):
        """
        这段代码实现了一个类的方法 __getitem__，它是 Python 中一个内置的方法，用于支持类的实例对象的索引操作。在这个方法中，首先调用了一个名为 lazy_init_from_ipc_handle 的方法来确保数据已经被加载到内存中。然后将输入的 node_idx 转换为与数据存储设备相同的设备，即 self.rank 所在的设备。

        接下来，如果数据没有被映射到内存中（即 self.mmap_handle_ 为 None），那么根据缓存策略以及特征顺序数组 self.feature_order 来选择从哪个设备上获取数据。如果缓存策略为 "device_replicate"，则从 self.device_tensor_list[self.rank] 中获取数据；否则从 self.clique_tensor_list[clique_id] 中获取数据，其中 clique_id 是 self.rank 所在的团的编号。在获取数据之前，还需根据特征顺序数组将 node_idx 的顺序调整为特征顺序。

        如果数据已经被映射到内存中，则需要根据 self.disk_map 找出哪些数据需要从磁盘中读取，哪些数据已经在内存中了。对于需要从磁盘中读取的数据，首先找出它们在 node_idx 中的位置和在磁盘中的编号，然后调用 self.read_mmap 方法读取数据。对于已经在内存中的数据，也需要根据 self.disk_map 找到它们在内存中的位置，然后从 self.device_tensor_list[self.rank] 或 self.clique_tensor_list[clique_id] 中获取数据。最后，根据 node_idx 在结果张量 res 中的位置，将从磁盘和内存中获取的数据分别放到 res 中，然后将 res 返回。

        Args:
            node_idx (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        self.lazy_init_from_ipc_handle()
        node_idx = node_idx.to(self.rank)
        if self.mmap_handle_ is None:
            if self.feature_order is not None:
                node_idx = self.feature_order[node_idx]
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list[self.rank]
                return shard_tensor[node_idx]
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list[clique_id]
                return shard_tensor[node_idx]
        else:
            num_nodes = node_idx.size(0)
            disk_index = self.disk_map[node_idx]
            node_range = torch.arange(end=num_nodes,
                                      device=self.rank,
                                      dtype=torch.int64)
            disk_mask = disk_index < 0
            mem_mask = disk_index >= 0
            disk_ids = torch.masked_select(node_idx, disk_mask)
            disk_pos = torch.masked_select(node_range, disk_mask)
            mem_ids = torch.masked_select(node_idx, mem_mask)
            mem_pos = torch.masked_select(node_range, mem_mask)
            local_mem_ids = self.disk_map[mem_ids]
            disk_res = self.read_mmap(disk_ids)
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list[self.rank]
                mem_res = shard_tensor[local_mem_ids]
            else:
                clique_id = self.topo.get_clique_id(self.rank)
                shard_tensor = self.clique_tensor_list[clique_id]
                mem_res = shard_tensor[local_mem_ids]
            res = torch.zeros((num_nodes, self.size(1)), device=self.rank)
            res[disk_pos] = disk_res
            res[mem_pos] = mem_res
            return res

    def size(self, dim: int):
        """ Get dim size for quiver.Feature

        Args:
            dim (int): dimension 

        Returns:
            int: dimension size for dim
        """
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.size(dim)
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            shard_tensor = self.clique_tensor_list[clique_id]
            return shard_tensor.size(dim)

    def dim(self):
        """ Get the number of dimensions for quiver.Feature

        Args:
            None

        Returns:
            int: number of dimensions
        """
        return len(self.shape)

    @property
    def shape(self):
        self.lazy_init_from_ipc_handle()
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.shape
        else:
            clique_id = self.topo.get_clique_id(self.rank)
            shard_tensor = self.clique_tensor_list[clique_id]
            return shard_tensor.shape

    @property
    def ipc_handle(self):
        return self.ipc_handle_

    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    def share_ipc(self):
        """Get ipc handle for multiprocessing

        Returns:
            tuples: ipc handles for ShardTensor and torch.Tensor and python native objects
        """
        self.cpu_part.share_memory_()
        gpu_ipc_handle_dict = {}
        if self.cache_policy == "device_replicate":
            for device in self.device_tensor_list:
                gpu_ipc_handle_dict[device] = self.device_tensor_list[
                    device].share_ipc()[0]
        else:
            for clique_id in self.clique_tensor_list:
                gpu_ipc_handle_dict[clique_id] = self.clique_tensor_list[
                    clique_id].share_ipc()[0]

        return gpu_ipc_handle_dict, self.cpu_part if self.cpu_part.numel() > 0 else None, self.device_list, self.device_cache_size, self.cache_policy, self.csr_topo

    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor):
        if self.cache_policy == "device_replicate":
            ipc_handle = gpu_ipc_handle_dict.get(
                self.rank, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
            self.device_tensor_list[self.rank] = shard_tensor

        else:
            clique_id = self.topo.get_clique_id(self.rank)
            ipc_handle = gpu_ipc_handle_dict.get(
                clique_id, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
            self.clique_tensor_list[clique_id] = shard_tensor

        self.cpu_part = cpu_tensor

    @classmethod
    def new_from_ipc_handle(cls, rank, ipc_handle):
        """Create from ipc handle

        Args:
            rank (int): device rank for feature collection kernels to launch
            ipc_handle (tuple): ipc handle create from `share_ipc`

        Returns:
            [quiver.Feature]: created quiver.Feature
        """
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = ipc_handle
        feature = cls(rank, device_list, device_cache_size, cache_policy)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        if csr_topo is not None:
            feature.feature_order = csr_topo.feature_order.to(rank)
        feature.csr_topo = csr_topo
        return feature

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, _ = ipc_handle
        feature = cls(device_list[0], device_list, device_cache_size,
                      cache_policy)
        feature.ipc_handle = ipc_handle
        return feature

    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return

        self.rank = torch.cuda.current_device()
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        self.csr_topo = csr_topo
        if csr_topo is not None:
            self.feature_order = csr_topo.feature_order.to(self.rank)

        self.ipc_handle = None


class PartitionInfo:
    """PartitionInfo is the partitioning information of how features are distributed across nodes.
    It is mainly used for distributed feature collection, by DistFeature.

    Args:
        device (int): device for local feature partition
        host (int): host id for current node
        hosts (int): the number of hosts in the cluster
        global2host (torch.Tensor): global feature id to host id mapping
        replicate (torch.Tensor, optional): CSRTopo of the graph for feature reordering

    """

    def __init__(self, device, host, hosts, global2host, replicate=None):
        self.global2host = global2host.to(device)
        self.host = host
        self.hosts = hosts
        self.device = device
        self.size = self.global2host.size(0)
        self.replicate = None
        if replicate is not None:
            self.replicate = replicate.to(device)
        self.init_global2local()

    def init_global2local(self):
        total_range = torch.arange(end=self.size,
                                   device=self.device,
                                   dtype=torch.int64)
        self.global2local = torch.arange(end=self.size,
                                         device=self.device,
                                         dtype=torch.int64)
        for host in range(self.hosts):
            mask = self.global2host == host
            host_nodes = torch.masked_select(total_range, mask)
            host_size = host_nodes.size(0)
            if host == self.host:
                local_size = host_size
            host_range = torch.arange(end=host_size,
                                      device=self.device,
                                      dtype=torch.int64)
            self.global2local[host_nodes] = host_range
        if self.replicate is not None:
            self.global2host[self.replicate] = self.host
            replicate_range = torch.arange(start=local_size,
                                           end=local_size +
                                           self.replicate.size(0),
                                           device=self.device,
                                           dtype=torch.int64)
            self.global2local[self.replicate] = replicate_range

    def dispatch(self, ids):
        host_ids = []
        host_orders = []
        ids_range = torch.arange(end=ids.size(0),
                                 dtype=torch.int64,
                                 device=self.device)
        host_index = self.global2host[ids]
        for host in range(self.hosts):
            mask = host_index == host
            host_nodes = torch.masked_select(ids, mask)
            host_order = torch.masked_select(ids_range, mask)
            host_nodes = self.global2local[host_nodes]
            host_ids.append(host_nodes)
            host_orders.append(host_order)
        torch.cuda.current_stream().synchronize()

        return host_ids, host_orders


class DistFeature:
    """DistFeature stores local features and it can fetch remote features by the network.
    Normally, each trainer process holds a DistFeature object. 
    We can create DistFeature by a local feature object, a partition information object and a network communicator.
    After creation, each worker process can collect features just like a local tensor.
    It is a synchronous operation, which means every process should collect features at the same time.

    ```python
    >>> info = quiver.feature.PartitionInfo(...)
    >>> comm = quiver.comm.NcclComm(...)
    >>> quiver_feature = quiver.Feature(...)
    >>> dist_feature = quiver.feature.DistFeature(quiver_feature, info, comm)
    >>> features = dist_feature[node_idx]
    ```

    Args:
        feature (Feature): local feature
        info (PartitionInfo): partitioning information across nodes
        comm (quiver.comm.NcclComm): communication topology for distributed features

    """

    def __init__(self, feature, info, comm):
        self.feature = feature
        self.info = info
        self.comm = comm

    def __getitem__(self, ids):
        ids = ids.to(self.comm.device)
        host_ids, host_orders = self.info.dispatch(ids)
        host_feats = self.comm.exchange(host_ids, self.feature)
        feats = torch.zeros((ids.size(0), self.feature.size(1)),
                            device=self.comm.device)
        for feat, order in zip(host_feats, host_orders):
            if feat is not None and order is not None:
                feats[order] = feat
        local_ids, local_order = host_ids[self.info.host], host_orders[
            self.info.host]
        feats[local_order] = self.feature[local_ids]
        return feats
