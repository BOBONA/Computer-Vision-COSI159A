import torch
from torch import cuda
from torch.nn import Module, CosineSimilarity
from torch.optim import Optimizer
from torch.utils.data import DataLoader

device = 'cuda' if cuda.is_available() else 'cpu'


def train(dataloader: DataLoader, model: Module, optimizer: Optimizer) -> float:
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(dataloader)
    return total_loss


def test(dataloader: DataLoader, model: Module) -> float:
    model.eval()
    sim_func = CosineSimilarity()

    correct = 0
    with torch.no_grad():
        for x, y, same in dataloader:
            x, y, same = x.to(device), y.to(device), same.to(device)
            feature_x = model.forward(x)
            feature_y = model.forward(y)
            similarity = sim_func(feature_x, feature_y)
            correct += sum(same == (similarity > 0.5))

    avg_correct = correct / len(dataloader.dataset)
    return avg_correct
