import argparse

parser = argparse.ArgumentParser(prog="SphereFace Training")
parser.add_argument('--epochs', type=int, default=128,
                    help="Number of epochs to train for")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for the optimizer")
parser.add_argument('--batch-size', type=int, default=128,
                    help="Batch size for the dataloader")
parser.add_argument('--load-model', type=bool, default=False,
                    help="Load the model from the model-path")
parser.add_argument('--model-path', type=str, default="model.pth",
                    help="Path to save the model")
args = parser.parse_args()

epochs: int = args.epochs
lr: float = args.lr
batch_size: int = args.batch_size
load_model: bool = args.load_model
model_path: str = args.model_path
