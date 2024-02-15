import torch
from torch import cuda
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

device = 'cuda' if cuda.is_available() else 'cpu'


def train(dataloader: DataLoader, model: Module, loss_fn: Module, optimizer: Optimizer):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader: DataLoader, model: Module, loss_fn: Module) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            test_loss += loss_fn(prediction, y)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    loss = test_loss / len(dataloader)
    avg_correct = correct / len(dataloader.dataset)
    return loss, avg_correct
