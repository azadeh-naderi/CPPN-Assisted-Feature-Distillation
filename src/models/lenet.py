import torch.nn as nn


class LeNet(nn.Module):
    """Ported from legacy/test_cppn_lenet.py. The `pretrained` branch called
    a `load_pretrained_weights()` method that was never defined anywhere in
    the original script (a dormant AttributeError, unreachable since the
    script always ran with pretrained=False) — dropped here rather than
    ported."""

    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        feature_map_size = 5 if input_channels == 1 else 6
        self.fc1 = nn.Linear(16 * feature_map_size * feature_map_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x, return_features: bool = False):
        x = self.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        features = self.sigmoid(self.fc2(x))
        logits = self.fc3(features)
        if return_features:
            return logits, features
        return logits
