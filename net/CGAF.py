import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpaCNN(nn.Module):
    def __init__(self, in_channels, SK_size=3, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size,
                               padding=SK_size//2, stride=strides)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size,
                               padding=SK_size//2, stride=strides)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y)

class SpatialCNNAtt(nn.Module):
    def __init__(self, in_channels=64, SK_size=3, kernel_size=3):
        super(SpatialCNNAtt, self).__init__()
        self.scnn = SpaCNN(in_channels=in_channels, SK_size=SK_size)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CGAF_Core(nn.Module):
    def __init__(self, channel=64, SK_size=3):
        super(CGAF_Core, self).__init__()
        self.spaCnnAtt = SpatialCNNAtt(in_channels=channel, SK_size=SK_size)
        self.chaAtt = ChannelAtt(in_channels=channel)
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)
    def forward(self, x, y):
        f = self.chaAtt(x) * self.spaCnnAtt(y)        # [B,C,H,W]
        z = torch.cat([f * x, f * y], dim=1)          # [B,2C,H,W]
        return F.relu(self.bn1(self.conv1(z)))        # [B,C,H,W]

class CGAF(CGAF_Core):
    def __init__(self, channel=64, SK_size=3,
                 gate_init: float = 1.0,
                 dropout_p: float = 0.0,
                 with_affine: bool = True):
        super().__init__(channel=channel, SK_size=SK_size)
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.with_affine = bool(with_affine)
        if self.with_affine:
            self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1))
            self.beta  = nn.Parameter(torch.zeros(1, channel, 1, 1))
        else:
            self.register_parameter("alpha", None)
            self.register_parameter("beta", None)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    @staticmethod
    def _apply_affine(x, alpha, beta):
        if alpha is not None and beta is not None:
            return alpha * x + beta
        return x

    def forward(self, x, y):
        out = super().forward(x, y)
        out = self._apply_affine(out, self.alpha, self.beta)
        out = self.dropout(out)
        out = self.gate * out
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)

    def freeze_affine_gate(self):
        for n, p in self.named_parameters():
            if any(k in n for k in ["gate", "alpha", "beta"]):
                p.requires_grad = False
        return self

