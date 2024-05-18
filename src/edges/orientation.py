import torch
from torch.nn import Module, Conv2d


class Orientation(Module):
    def __init__(self, r):
        super().__init__()
        kernel_size = 2 * r + 1
        kernel = torch.Tensor(list(range(1, r + 1)) + [r + 1] + list(range(r, 0, -1)))
        self.conv1 = Conv2d(
            1, 1, (1, kernel_size), padding="same", padding_mode="reflect", bias=False
        )
        self.conv1.weight.data = kernel.reshape((1, 1, 1, kernel_size))
        self.conv2 = Conv2d(
            1, 1, (kernel_size, 1), padding="same", padding_mode="reflect", bias=False
        )
        self.conv2.weight.data = kernel.reshape((1, 1, kernel_size, 1))

    def forward(self, x):
        self.eval()
        ori_y, ori_x = torch.gradient(self.conv2(self.conv1(x)), dim=[2, 3])
        (ori_xx,) = torch.gradient(ori_x, dim=3)
        ori_yy, ori_xy = torch.gradient(ori_y, dim=[2, 3])
        ori = torch.atan(ori_yy * torch.sign(-ori_xy) / (ori_xx + 1e-5)) % torch.pi
        return ori
