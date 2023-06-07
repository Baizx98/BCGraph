# baseline 模型训练的指标测试
# pyg单GPU reddit数据集 loss曲线，准确率曲线，数据加载和计算时间占比
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from graphsage import GraphSAGE
from tqdm import tqdm


def pyg_1gpu_reddit():
    path = "/home8t/bzx/data/Reddit"
    dataset = Reddit(path)
    data = dataset[0]
    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=data.train_mask,
        sizes=[25, 10],
        batch_size=1024,
        shuffle=True,
        num_workers=12,
    )
    subgraph_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=1024,
        shuffle=False,
        num_workers=12,
    )
    writer = SummaryWriter("./profiler/data/log")

    device = torch.device("cuda:1")
    model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    def train(epoch):
        model.train()
        pbar = tqdm(total=int(data.train_mask.sum()))
        pbar.set_description(f"Epoch {epoch:02d}")

        total_loss = total_correct = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            pbar.update(batch_size)

        pbar.close()

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / int(data.train_mask.sum())
        return loss, approx_acc

    @torch.no_grad()
    def test():
        model.eval()

        out = model.inference(x, device, subgraph_loader)

        y_true = y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        results = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

        return results

    for epoch in range(1, 100):
        loss, acc = train(epoch)
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("acc", acc, epoch)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")
        train_acc, val_acc, test_acc = test()
        writer.add_scalars(
            "eva_acc",
            {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc},
            epoch,
        )

        print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, " f"Test: {test_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    pyg_1gpu_reddit()
