from torch import diagonal, sum, exp, mean, log, cat, cos, acos, clamp
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import relu, normalize


class AngularPenaltySMLoss(Module):
    def __init__(self, in_features, out_features, eps=1e-7, m=4):
        """
        Angular Penalty Softmax Loss

        Credit to github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/
        """

        super(AngularPenaltySMLoss, self).__init__()

        self.m = m
        self.eps = eps
        self.fc = Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        """
        Input shape (N, in_features)
        """

        x = normalize(x, p=2, dim=1)

        wf = self.fc(x)
        numerator = cos(self.m * acos(
            clamp(diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = cat([cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = exp(numerator) + sum(exp(excl), dim=1)
        loss = numerator - log(denominator)
        return -mean(loss)


class Sphereface4Layer(Module):
    def __init__(self, class_num):
        super(Sphereface4Layer, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2)
        self.fc = Linear(512 * 5 * 5, 512)
        self.ang = AngularPenaltySMLoss(512, class_num)

    def forward(self, x, y=None):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = x.view(-1, 512 * 5 * 5)
        x = self.fc(x)

        # only return the loss if a label is provided
        if y is not None:
            ang = self.ang(x, y)
            return x, ang
        else:
            return x
