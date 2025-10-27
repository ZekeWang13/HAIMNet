import torch
import torch.nn as nn
import torch.nn.functional as F

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class IGAM(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, dropout_p: float = 0.0, gate_init: float = 1.0):
        super().__init__()
        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

        self.gate_smfa = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.gate_pcfn = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

        self.alpha_smfa = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta_smfa  = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.alpha_pcfn = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta_pcfn  = nn.Parameter(torch.zeros(1, dim, 1, 1))

        self.drop_smfa = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.drop_pcfn = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    @staticmethod
    def _norm(x):
        return F.normalize(x)

    def forward(self, x):
        res1 = self.smfa(self._norm(x))
        res1 = self.alpha_smfa * res1 + self.beta_smfa
        res1 = self.drop_smfa(res1)
        x = x + self.gate_smfa * res1

        res2 = self.pcfn(self._norm(x))
        res2 = self.alpha_pcfn * res2 + self.beta_pcfn
        res2 = self.drop_pcfn(res2)
        x = x + self.gate_pcfn * res2
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)