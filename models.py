from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_features = 561
        self.n_classes = 12

        self.linear_1 = nn.Linear(self.n_features, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x
