import torch
from torch import nn, cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Grayscale

from models import BasicMNISTModel, ResNetMNISTModel
from training import train, test
import args

device = 'cuda' if cuda.is_available() else 'cpu'


def main_routine():
    writer = SummaryWriter()

    transform = ToTensor() if args.model == "basic" else Compose([Grayscale(num_output_channels=3), ToTensor()])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    print("Training on", device)
    model = (BasicMNISTModel() if args.model == "basic"
             else ResNetMNISTModel(pretrained_weights=args.load_pretrained,
                                   freeze_all_but_fc=args.freeze_all_but_fc)).to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_avg_correct = test(test_dataloader, model, loss_fn)
        training_loss, training_avg_correct = test(train_dataloader, model, loss_fn)

        writer.add_scalar('Loss', test_loss, epoch)
        writer.add_scalar('Accuracy', test_avg_correct, epoch)
        writer.add_scalar('Training Accuracy', training_avg_correct, epoch)
        torch.save(model.state_dict(), args.model_path)

        print("Epoch", epoch + 1, "of", args.epochs)

    writer.close()


if __name__ == '__main__':
    main_routine()
