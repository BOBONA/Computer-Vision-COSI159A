# Assignment 2

This assignment explores the effect of the angular loss used in the SphereFace model. The code allows for basic training and testing. PyTorch is used for machine learning and Tensorboard is used for visualization. The default parameters achieve a testing accuracy of 74% after 128 epochs.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages. Note that the PyTorch version might need to be modified to fit your environment.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py [--epochs EPOCHS] [--lr LR] [--batch-size BATCH_SIZE]
               [--load-model LOAD_MODEL] [--model-path MODEL_PATH]
```

### Arguments

- `--epochs` (int, default: 30): Number of epochs to train the model.

- `--lr` (float, default: 0.001): Learning rate for the optimizer.

- `--batch-size` (int, default: 64): Batch size for the data loader.

- `--load-model` (bool, default: False): Whether to load a previously saved model. If `True`, specify the model path.

- `--model-path` (str, default: "mnist.pth"): Path to save the trained model.

### Examples

1. Train a model with default settings:
    ```
    python main.py
    ```

2. Train a model with custom settings:
    ```
    python main.py --epochs 20 --lr 0.0005 --batch-size 32
    ```

3. Load a previously saved model for further training:
    ```
    python main.py --load-model True --model-path saved_model.pth
    ```

### Notes

- If `--load-model` is set to `True`, provide the path to the saved model using `--model-path`.

### Tensorboard

The script logs the training and validation loss and accuracy to the `runs` directory. To visualize the training process, run the following command in the terminal:

```bash
tensorboard --logdir runs
```

## Report

The assignment report is included in the repository as a [PDF file](submission.pdf).