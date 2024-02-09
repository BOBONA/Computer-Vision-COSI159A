import torch
from torch import nn, cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from models import BasicMNISTModel
from training import train, test

device = 'cuda' if cuda.is_available() else 'cpu'

epochs = 10
lr = 0.01
batch_size = 32


def main():
    writer = SummaryWriter()

    train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = BasicMNISTModel().to(device)
    model.load_state_dict(torch.load('mnist.pth'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)

    for epoch in range(epochs):
        print("Epoch", epoch)

        train(train_dataloader, model, loss_fn, optimizer)
        loss, avg_correct = test(test_dataloader, model, loss_fn, epoch)

        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Accuracy', avg_correct, epoch)
        torch.save(model.state_dict(), 'mnist.pth')

    writer.close()


if __name__ == '__main__':
    main()
