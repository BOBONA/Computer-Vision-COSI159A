from torch.nn import Module, Conv2d, Dropout2d, Linear
from torch.nn.functional import relu, max_pool2d, dropout, log_softmax


class BasicMNISTModel(Module):
    def __init__(self):
        super(BasicMNISTModel, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = Dropout2d()
        self.fc1 = Linear(320, 50)
        self.fc2 = Linear(50, 10)

    def forward(self, x):
        x = relu(max_pool2d(self.conv1(x), 2))
        x = relu(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = relu(self.fc1(x))
        x = dropout(x, training=self.training)
        x = self.fc2(x)
        return log_softmax(x, dim=1)