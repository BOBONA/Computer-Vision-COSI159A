# Assignment 1

This assignment explores training classifiers on MNIST. The code allows for training with a basic convolutional network and with ResNet50. PyTorch is used for machine learning and Tensorboard is used for visualization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages. Note that the PyTorch version might need to be modified to fit your environment.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py [--model MODEL] [--epochs EPOCHS] [--lr LR] [--batch-size BATCH_SIZE]
               [--freeze-all-but-fc FREEZE_ALL_BUT_FC] [--load-pretrained LOAD_PRETRAINED]
               [--load-model LOAD_MODEL] [--model-path MODEL_PATH]
```

### Arguments

- `--model` (str, default: "resnet"): Specifies the model architecture to use. Choose between "basic" and "resnet".

- `--epochs` (int, default: 30): Number of epochs to train the model.

- `--lr` (float, default: 0.001): Learning rate for the optimizer.

- `--batch-size` (int, default: 64): Batch size for the data loader.

- `--freeze-all-but-fc` (bool, default: False): If using ResNet, this flag freezes all layers except the fully connected layer.

- `--load-pretrained` (bool, default: True): Whether to load pretrained weights for the ResNet model.

- `--load-model` (bool, default: False): Whether to load a previously saved model. If `True`, specify the model path.

- `--model-path` (str, default: "mnist.pth"): Path to save the trained model.

### Examples

1. Train a ResNet model with default settings:
    ```
    python main.py
    ```

2. Train a basic model with custom settings:
    ```
    python main.py --model basic --epochs 20 --lr 0.0005 --batch-size 32
    ```

3. Train a ResNet model, freezing all layers except the fully connected layer:
    ```
    python main.py --freeze-all-but-fc True
    ```

4. Load a pretrained ResNet model and train for more epochs:
    ```
    python main.py --load-pretrained True --epochs 50
    ```

5. Load a previously saved model for further training:
    ```
    python main.py --load-model True --model-path saved_model.pth
    ```

### Notes

- For the `--freeze-all-but-fc` option, only applicable when the ResNet architecture is chosen.

- The `--load-pretrained` option is only effective when using the ResNet model.

- If `--load-model` is set to `True`, provide the path to the saved model using `--model-path`.

### Tensorboard

The script logs the training and validation loss and accuracy to the `runs` directory. To visualize the training process, run the following command in the terminal:

```bash
tensorboard --logdir runs
```

## Report

The assignment report is included in the repository as a [PDF file](submission.pdf).