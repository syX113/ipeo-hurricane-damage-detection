import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Lightweight CNN baseline which uses 4 downsampling blocks followed by global pooling + MLP head.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, p_drop=0.05),
            ConvBlock(32, 64, p_drop=0.05),
            ConvBlock(64, 128, p_drop=0.05),
            ConvBlock(128, 256, p_drop=0.05),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        head = [
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
