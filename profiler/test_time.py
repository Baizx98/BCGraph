import os.path as osp
import os
import time
from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


import quiver


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

        pbar.close()

        return x_all


dataset_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Reddit")

dataset = Reddit(dataset_path)
data: Data = dataset[0]
train_idx: torch.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)

train_loader = torch.utils.data.DataLoader(
    train_idx, batch_size=1024, shuffle=True, drop_last=True
)  # Quiver
csr_topo = quiver.CSRTopo(data.edge_index)  # Quiver
quiver_sampler = quiver.pyg.GraphSageSampler(
    csr_topo, sizes=[25, 10], device=0, mode="GPU"
)  # Quiver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


x = quiver.Feature(
    rank=0,
    device_list=[0],
    device_cache_size="0.5G",
    cache_policy="device_replicate",
    csr_topo=csr_topo,
)  # Quiver
x.from_cpu_tensor(data.x)  # Quiver
y = data.y.squeeze().to(device)
print(data.x.element_size())
print(data.x.size())

count = 0
batch_elapsed = 0


def train(epoch):
    model.train()
    global count
    global batch_elapsed
    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    ############################################
    # Step 3: Training the PyG Model with Quiver
    ############################################
    # for batch_size, n_id, adjs in train_loader: # Original PyG Code
    with torch.profiler.profile(
        profile_memory=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/data/logs"),
    ) as prof:
        for seeds in train_loader:  # Quiver
            count = count + 1
            batch_start = time.time()
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)  # Quiver
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            batch_end = time.time()
            batch_elapsed = batch_end - batch_start + batch_elapsed
            pbar.update(batch_size)
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


epoch_elapsed = 0

for epoch in range(1, 11):
    epoch_start = time.time()
    loss, acc = train(epoch)
    epoch_end = time.time()
    epoch_elapsed = epoch_end - epoch_start + epoch_elapsed
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")
print("每个epoch平均计算时间为：", epoch_elapsed / 10, " s")
print("每个batch平均计算时间为：", batch_elapsed / count, " s")
