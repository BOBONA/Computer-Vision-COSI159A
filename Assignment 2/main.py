import time

import torch
from torch import cuda
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import LFWPeople, LFWPairs

from models import Sphereface4Layer
from training import train, test
import args

device = 'cuda' if cuda.is_available() else 'cpu'

debug = True

if debug:
    from torch.utils.tensorboard import SummaryWriter  # a little hack to deal with issues with the debugger


def main_routine():
    if debug:
        writer = SummaryWriter()

    # some basic data augmentation
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
    scheduler = lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)  # reduce learning rate by 10 every 64 epochs

    for epoch in range(args.epochs):
        t1 = time.time()
        total_loss = train(train_dataloader, model, optimizer)
        scheduler.step()

        print("Epoch: %s; Loss: %.3f; Time: %.3f" % (str(epoch + 1).zfill(2), total_loss, time.time() - t1))
        if debug:
            writer.add_scalar('Loss', total_loss, epoch)

        # only evaluate every 5 epochs to save some time
        if (epoch + 1) % 5 == 0:
            t2 = time.time()
            test_avg_correct = test(test_dataloader, model)
            print("Acc.: %.4f; Time: %.3f" % (test_avg_correct, time.time() - t2))
            if debug:
                writer.add_scalar('Accuracy', test_avg_correct, epoch)

        torch.save(model.state_dict(), args.model_path)

    if debug:
        writer.close()


if __name__ == '__main__':
    main_routine()
