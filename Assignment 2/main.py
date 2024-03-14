import torch
from torch import nn, cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import LFWPeople, LFWPairs

from models import Sphereface4Layer
from training import train, test
import args

device = 'cuda' if cuda.is_available() else 'cpu'


def main_routine():
    writer = SummaryWriter()

    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = LFWPeople(root='./data', split='Train', download=False, transform=transform)
    test_dataset = LFWPairs(root='./data', split='Test', download=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    class_count = len(train_dataset.class_to_idx)

    print("Training on", device)
    model = Sphereface4Layer(class_count).to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
    optimizer = Adam(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        total_loss = train(train_dataloader, model, optimizer)
        test_avg_correct = test(test_dataloader, model)

        writer.add_scalar('Loss', total_loss, epoch)
        writer.add_scalar('Accuracy', test_avg_correct, epoch)
        torch.save(model.state_dict(), args.model_path)

        print("Epoch", epoch + 1, "of", args.epochs)

    writer.close()


if __name__ == '__main__':
    main_routine()
