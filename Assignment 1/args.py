import argparse

parser = argparse.ArgumentParser(prog="MNIST Training")
parser.add_argument("--model", type=str, choices=["basic", "resnet"], default="resnet",
                    help="Model architecture to use (either 'basic' or 'resnet')")
parser.add_argument("--epochs", type=int, default=30,
                    help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate for the optimizer")
parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for the dataloader")
parser.add_argument("--freeze-all-but-fc", type=bool, default=False,
                    help="For resnet, freeze all but the fully connected layer")
parser.add_argument("--load-pretrained", type=bool, default=True,
                    help="Load the pretrained weights for the resnet model")
parser.add_argument("--load-model", type=bool, default=False,
                    help="Load the model from the model-path")
parser.add_argument("--model-path", type=str, default="mnist.pth",
                    help="Path to save the model")
args = parser.parse_args()

model: str = args.model
epochs: int = args.epochs
lr: float = args.lr
batch_size: int = args.batch_size
freeze_all_but_fc: bool = args.freeze_all_but_fc
load_pretrained: bool = args.load_pretrained
load_model: bool = args.load_model
model_path: str = args.model_path
