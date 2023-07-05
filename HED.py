import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # 定义卷积核参数
        k1 = torch.tensor([[1.88, -1.02, -0.55]])
        k2 = torch.tensor([[-0.07, 1.13, -0.13]])
        k3 = torch.tensor([[-0.60, -0.48, 1.57]])

        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)

        # 将卷积核参数设置为定义的值
        self.conv1.weight = torch.nn.Parameter(k1.unsqueeze(2).unsqueeze(3))
        self.conv2 .weight = torch.nn.Parameter(k2.unsqueeze(2).unsqueeze(3))
        self.conv3.weight = torch.nn.Parameter(k3.unsqueeze(2).unsqueeze(3))

        # Define batch normalization layer
        self.batch_norm = nn.BatchNorm2d(num_features=3)

    def forward(self, x):
        print(x)
        # Perform convolutions
        h = self.conv1(x)
        print(h)
        e = self.conv2(x)
        print(e)
        d = self.conv3(x)
        print(d)

        # Add the results together
        hed = torch.cat([h, e, d], dim=1)
        print(hed)

        # Apply batch normalization
        print(x)
        out = self.batch_norm(hed)
        x = x + out
        print(x)

        return x

model = CustomModel()
input_tensor = torch.randn(1, 3, 2, 2)  # batch size = 2, height = 32, width = 32
output_tensor = model(input_tensor)
print(output_tensor.shape)  # prints torch.Size([2, 3, 32, 32])
