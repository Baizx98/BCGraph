import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch_quiver as qv
from typing import List, Tuple, NamedTuple, Generic, TypeVar

from dataclasses import dataclass
import torch.multiprocessing as mp
import itertools
import time
import quiver
import quiver.utils as quiver_utils


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

__all__ = ["GraphSageSampler", "MixedGraphSageSampler", "SampleJob"]


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        """将对象转移到指定设备上"""
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


@dataclass(frozen=True)
class _FakeDevice(object):
    pass


@dataclass(frozen=True)
class _StopWork(object):
    pass


class GraphSageSampler:
    r"""
    Quiver's GraphSageSampler behaves just like Pyg's `NeighborSampler` but with much higher performance.
    It can work in `UVA` mode or `GPU` mode. You can set `mode=GPU` if you have enough GPU memory to place graph's topology data which will offer the best sample performance.
    When your graph is too big for GPU memory, you can set `mode=UVA` to still use GPU to perform sample but place the data in host memory. `UVA` mode suffers 30%-40% performance loss compared to `GPU` mode
    but is much faster than CPU sampling(normally 16x~20x) and it consumes much less GPU memory compared to `GPU` mode.

    Args:
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`UVA`, `GPU`, `CPU`], default is `UVA`.
    """

    def __init__(self,
                 csr_topo: quiver_utils.CSRTopo,
                 sizes: List[int],
                 device=0,
                 mode="UVA"):

        assert mode in ["UVA",
                        "GPU",
                        "CPU"], f"sampler mode should be one of [UVA, GPU]"
        assert device is _FakeDevice or mode == "CPU" or (
            device >= 0 and mode != "CPU"), f"Device setting and Mode setting not compatitive"

        self.sizes = sizes
        self.quiver = None
        self.csr_topo = csr_topo
        self.mode = mode
        if self.mode in ["GPU", "UVA"] and device is not _FakeDevice and device >= 0:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                          self.csr_topo.indices,
                                                          edge_id, device,
                                                          self.mode != "UVA")
        elif self.mode == "CPU" and device is not _FakeDevice:
            self.quiver = qv.cpu_quiver_from_csr_array(
                self.csr_topo.indptr, self.csr_topo.indices)
            device = "cpu"

        self.device = device
        self.ipc_handle_ = None

    def sample_layer(self, batch, size):
        """
        该函数的主要作用是从给定的节点批次(batch)中，随机采样邻居节点。其实现流程如下：

        调用lazy_init_quiver()函数，如果尚未初始化，将初始化Quiver图数据结构，即根据csr_topo.indptr和csr_topo.indices创建一个Quiver对象。
        将传入的节点批次(batch)转换为torch.tensor类型（如果不是的话）。
        将节点批次(batch)转移到GPU上，如果是GPU模式。
        根据传入的size参数（如果size为-1，则返回整个节点集合），使用Quiver图数据结构进行随机采样邻居节点。
        返回采样后的邻居节点和其数量。
        如果模式是GPU或UVA，则会使用Quiver的sample_neighbor方法来采样邻居节点，该方法需要传入批次号batch和采样节点的数量size，返回采样后的节点和节点数量。否则，将直接使用sample_neighbor方法采样邻居节点，该方法需要传入节点ID和采样节点的数量size，同样返回采样后的节点和节点数量。

        需要注意的是，在GPU或UVA模式下，节点ID和批次号都必须转移到设备上。同时，由于采样邻居节点需要访问Quiver图数据结构，因此需要调用lazy_init_quiver()函数来确保Quiver图数据结构已初始化。
        """
        self.lazy_init_quiver()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(self.device)
        size = size if size != -1 else self.csr_topo.node_count
        if self.mode in ["GPU", "UVA"]:
            n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        else:
            n_id, count = self.quiver.sample_neighbor(n_id, size)

        return n_id, count

    def lazy_init_quiver(self):
        """懒初始化quiver 为什么要这样做"""

        if self.quiver is not None:
            return

        self.device = "cpu" if self.mode == "CPU" else torch.cuda.current_device()

        if "CPU" == self.mode:
            self.quiver = qv.cpu_quiver_from_csr_array(
                self.csr_topo.indptr, self.csr_topo.indices)
        else:
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                          self.csr_topo.indices,
                                                          edge_id, self.device,
                                                          self.mode != "UVA")

    def reindex(self, inputs, outputs, counts):
        return self.quiver.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        """Sample k-hop neighbors from input_nodes

        这是一个Python函数，用于从输入节点中采样k-hop邻居，并返回结果，其中k是预先定义的不同采样层数的列表。函数的具体步骤如下：

        首先，函数调用了lazy_init_quiver()方法，以确保Quiver实例已经初始化。
        接下来，函数将输入节点input_nodes转换为self.device所指示的设备（通常是GPU）上的Tensor。
        然后，函数对于每个指定的采样层数size，调用了sample_layer()方法来采样邻居节点，并将结果存储在一个列表adjs中。
        对于每个采样层，函数将采样得到的邻居节点重排（reindex）并构造成邻接矩阵的形式（即edge_index、row_idx、col_idx），并将结果封装在一个Adj对象中，并将该对象添加到adjs列表中。同时，函数还将下一层采样的输入节点设置为当前层采样得到的邻居节点。
        最后，函数将采样得到的最后一层节点和构造的邻接矩阵列表返回。需要注意的是，由于adjs列表是按照采样层数从小到大存储的，因此需要将其反转（使用[::-1]）以便返回的邻接矩阵按照采样层数从大到小排列。

        Args:
            input_nodes (torch.LongTensor): seed nodes ids to sample from

        Returns:
            Tuple: Return results are the same with Pyg's sampler
        """
        self.lazy_init_quiver()

        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
        for size in self.sizes:
            out, cnt = self.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            adj_size = torch.LongTensor([
                frontier.size(0),
                nodes.size(0),
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, adj_size))
            nodes = frontier

        return nodes, batch_size, adjs[::-1]

    def sample_prob(self, train_idx, total_node_count):
        self.lazy_init_quiver()
        last_prob = torch.zeros(total_node_count, device=self.device)
        last_prob[train_idx] = 1
        for size in self.sizes:
            cur_prob = torch.zeros(total_node_count, device=self.device)
            self.quiver.cal_neighbor_prob(0, last_prob, cur_prob, size)
            last_prob = cur_prob
        return last_prob

    def share_ipc(self):
        """Create ipc handle for multiprocessing

        Returns:
            tuple: ipc handle tuple
        """
        return self.csr_topo, self.sizes, self.mode

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        """Create from ipc handle

        Args:
            ipc_handle (tuple): ipc handle got from calling `share_ipc`

        Returns:
            quiver.pyg.GraphSageSampler: Sampler created from ipc handle
        """
        csr_topo, sizes, mode = ipc_handle
        return cls(csr_topo, sizes, _FakeDevice, mode)


class SampleJob(Generic[T_co]):
    """
    An abstract class representing a :class:`SampleJob`.
    All SampleJobs that represent a map from index to sample tasks should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, :meth:`__getitem__`, :meth:`shuffle`, 
    supporting fetching a sample task for a given index, return the size of the SampleJob and shuffle all tasks in the Job.


    该代码定义了一个泛型类 SampleJob，它是一个抽象类（abstract class），不能被直接实例化。所有继承 SampleJob 的子类都必须实现三个方法：__getitem__、__len__ 和 shuffle。

    __getitem__(self, index) -> T_co：该方法用于获取指定索引 index 对应的样本任务，返回值类型为泛型变量 T_co，这个泛型变量表示该泛型类返回值的类型，而 -> 符号表示该方法的返回类型注解。
    __len__(self) -> int：该方法返回该泛型类中样本任务的总数，返回类型为整型。
    shuffle(self) -> None：该方法用于打乱样本任务的顺序，返回值为 None，即无返回值。
    该泛型类是一个表示样本任务集合的抽象基类，提供了对样本任务集合的一些基本操作，方便具体子类实现其它功能。
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def shuffle(self) -> None:
        raise NotImplementedError


def cpu_sampler_worker_loop(rank, quiver_sampler, task_queue, result_queue):
    """
    这段代码是一个CPU采样器的工作循环。主要流程如下：

    使用一个while循环来不断地获取任务，直到收到终止信号（_StopWork）为止。

    从任务队列task_queue中获取一个任务task，其中task是一个用于指定采样的任务。

    检查任务是否为终止信号，如果是，将终止信号加入结果队列result_queue中，并退出循环；否则，执行下一步操作。

    调用quiver_sampler对象的sample方法对任务进行采样，将采样结果res加入结果队列result_queue中。

    这段代码中，task_queue和result_queue分别是任务队列和结果队列，用于在主进程和采样器子进程之间传递任务和结果。quiver_sampler是一个采样器对象，用于实现采样的具体逻辑。rank参数表示采样器的进程编号。
    """
    while True:
        task = task_queue.get()
        if task is _StopWork:
            result_queue.put(_StopWork)
            break
        res = quiver_sampler.sample(task)
        result_queue.put(res)


class MixedGraphSageSampler:
    r"""
    Quiver's MixedGraphSageSampler behaves just like `GraphSageSampler` but trying to utilize both GPUs and CPUs to sample graphs.
    It can work in one of [`GPU_CPU_MIXED`, `UVA_CPU_MIXED`, `UVA_ONLY`, `GPU_ONLY`] modes. 
    Args:
        sample_job (SampleJob): A quiver.SampleJob for describing sample tasks
        num_workers (int): Decide parallelism for CPU sampler.
        csr_topo (quiver.CSRTopo): A quiver.CSRTopo for graph topology
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to `sizes[l] = -1`, all neighbors are included
            in layer `l`.
        device (int): Device which sample kernel will be launched
        mode (str): Sample mode, choices are [`GPU_CPU_MIXED`, `UVA_CPU_MIXED`, `UVA_ONLY`, `GPU_ONLY`], default is `UVA_CPU_MIXED`.
    """

    def __init__(self,
                 sample_job: SampleJob,
                 num_workers: int,
                 csr_topo: quiver_utils.CSRTopo,
                 sizes: List[int],
                 device=0,
                 mode="UVA_CPU_MIXED"):

        assert mode in ["UVA_CPU_MIXED", "GPU_CPU_MIXED", "UVA_ONLY",
                        "GPU_ONLY"], f"mode should be one of {['UVA_CPU_MIXED', 'GPU_CPU_MIXED', 'UVA_ONLY', 'GPU_ONLY']}"

        self.csr_topo = csr_topo
        self.csr_topo.share_memory_()

        self.device = device
        self.sample_job = sample_job
        self.num_workers = num_workers
        self.sizes = sizes
        self.mode = mode

        self.device_quiver = None
        self.cpu_quiver = None

        self.result_queue = None
        self.task_queues = []
        self.device_task_remain = None
        self.cpu_task_remain = None
        self.current_task_id = 0
        self.device_sample_time = 0
        self.cpu_sample_time = 0
        self.device_sample_total = 0
        self.cpu_sample_total = 0

        self.worker_ids = itertools.cycle(range(self.num_workers))

        self.inited = False
        self.epoch = 0

    def __iter__(self):
        self.sample_job.shuffle()
        if self.epoch <= 1:
            self.device_task_remain = None
            self.cpu_task_remain = None
            self.device_sample_time = 0
            self.cpu_sample_time = 0
            self.device_sample_total = 0
            self.cpu_sample_total = 0

        self.current_task_id = 0
        self.epoch += 1
        return self.iter_sampler()

    def decide_task_num(self):
        if self.device_task_remain is None:
            self.device_task_remain = max(20, self.num_workers * 2)
            if self.mode in ["GPU_ONLY", "UVA_ONLY"]:
                self.cpu_task_remain = 0
            else:
                self.cpu_task_remain = self.num_workers
        else:
            self.device_task_remain = max(20, self.num_workers * 2)
            if self.mode in ["GPU_ONLY", "UVA_ONLY"]:
                self.cpu_task_remain = 0
            else:
                self.cpu_task_remain = max(1, int(
                    self.device_sample_time * self.device_task_remain / self.cpu_sample_time / 2))

        print(
            f"Device average sample time: {self.device_sample_time}\tCPU average sample time: {self.cpu_sample_time}")
        print(
            f"Assign {self.device_task_remain} tasks to Device, Assign {self.cpu_task_remain} to CPU")

    def assign_cpu_tasks(self) -> bool:
        for task_id in range(self.current_task_id + self.device_task_remain, self.current_task_id + self.device_task_remain + self.cpu_task_remain):
            if task_id >= len(self.sample_job):
                break
            worker_id = next(self.worker_ids)
            self.task_queues[worker_id].put(self.sample_job[task_id])

    def lazy_init(self):

        if self.inited:
            return

        self.inited = True
        self.device_quiver = GraphSageSampler(
            self.csr_topo, self.sizes, device=self.device, mode="GPU" if "GPU" in self.mode else "UVA")
        self.cpu_quiver = GraphSageSampler(
            self.csr_topo, self.sizes, mode="CPU")
        self.result_queue = mp.Queue()
        for worker_id in range(self.num_workers):
            task_queue = mp.Queue()
            child_process = mp.Process(target=cpu_sampler_worker_loop,
                                       args=(worker_id, self.cpu_quiver, task_queue, self.result_queue))
            child_process.daemon = True
            child_process.start()
            self.task_queues.append(task_queue)

    def iter_sampler(self):
        self.lazy_init()
        try:

            while True:
                self.decide_task_num()
                self.assign_cpu_tasks()
                while self.device_task_remain > 0:
                    sample_start = time.time()
                    if self.current_task_id >= len(self.sample_job):
                        break

                    res = self.device_quiver.sample(
                        self.sample_job[self.current_task_id])
                    sample_end = time.time()

                    # Decide average sample time
                    self.device_sample_time = (
                        self.device_sample_time * self.device_sample_total + sample_end - sample_start) / (self.device_sample_total + 1)
                    self.device_sample_total += 1

                    self.current_task_id += 1
                    self.device_task_remain -= 1
                    yield res

                if self.current_task_id >= len(self.sample_job):
                    break
                while self.cpu_task_remain > 0:
                    sample_start = time.time()
                    res = self.result_queue.get()
                    sample_end = time.time()

                    # Decide average sample time
                    self.cpu_sample_time = (self.cpu_sample_time * self.cpu_sample_total +
                                            sample_end - sample_start) / (self.cpu_sample_total + 1)
                    self.cpu_sample_total += 1

                    self.current_task_id += 1
                    self.cpu_task_remain -= 1

                    if self.current_task_id >= len(self.sample_job):
                        break

                    yield res

                # Decide to exit
                if self.current_task_id >= len(self.sample_job):
                    break
        except:
            print("something wrong")
            # make sure all child process exit
            for task_queue in self.task_queues:
                task_queue.put(_StopWork)

            for _ in self.task_queues:
                self.result_queue.get()

    def share_ipc(self):
        return self.sample_job, self.num_workers, self.csr_topo, self.sizes, self.device, self.mode

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        sample_job, num_workers, csr_topo, sizes, device, mode = ipc_handle
        return cls(sample_job, num_workers, csr_topo, sizes, device, mode)
