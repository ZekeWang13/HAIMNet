import weakref
import torch.nn as nn

def _set_hidden_attr(obj, name, value):
    object.__setattr__(obj, name, value)

class IAMB(nn.Module):
    def __init__(self, owner_module: nn.Module,
                 i_lca_name: str, hv_lca_name: str,
                 igam_i_name: str, igam_h_name: str):
        super().__init__()
        _set_hidden_attr(self, "_owner_ref", weakref.ref(owner_module))
        self.i_lca_name = i_lca_name
        self.hv_lca_name = hv_lca_name
        self.igam_i_name = igam_i_name
        self.igam_h_name = igam_h_name

    def forward(self, i_in, hv_in):
        owner = self._owner_ref()
        I_LCA  = getattr(owner, self.i_lca_name)
        HV_LCA = getattr(owner, self.hv_lca_name)
        IGAM_i = getattr(owner, self.igam_i_name)
        IGAM_h = getattr(owner, self.igam_h_name)

        i_lca = I_LCA(i_in, hv_in)
        hv_lca = HV_LCA(hv_in, i_in)
        i_out = IGAM_i(i_lca)
        hv_out = IGAM_h(hv_lca)
        return i_out, hv_out


class IAMB_CGAF(nn.Module):
    def __init__(self, owner_module: nn.Module,
                 i_lca_name: str, hv_lca_name: str,
                 cgaf_i_name: str, cgaf_h_name: str,
                 igam_i_name: str, igam_h_name: str):
        super().__init__()
        _set_hidden_attr(self, "_owner_ref", weakref.ref(owner_module))
        self.i_lca_name = i_lca_name
        self.hv_lca_name = hv_lca_name
        self.cgaf_i_name = cgaf_i_name
        self.cgaf_h_name = cgaf_h_name
        self.igam_i_name = igam_i_name
        self.igam_h_name = igam_h_name

    def forward(self, i_in, hv_in):
        owner = self._owner_ref()
        I_LCA  = getattr(owner, self.i_lca_name)
        HV_LCA = getattr(owner, self.hv_lca_name)
        CGAF_i = getattr(owner, self.cgaf_i_name)
        CGAF_h = getattr(owner, self.cgaf_h_name)
        IGAM_i = getattr(owner, self.igam_i_name)
        IGAM_h = getattr(owner, self.igam_h_name)

        i4 = I_LCA(i_in, hv_in)
        h4 = HV_LCA(hv_in, i_in)
        i_c = CGAF_i(i4, h4)
        h_c = CGAF_h(h4, i4)
        i_out = IGAM_i(i_c)
        h_out = IGAM_h(h_c)
        return i_out, h_out


class IAMBup(nn.Module):
    def __init__(self, owner_module: nn.Module,
                 igam_i_name: str, igam_h_name: str,
                 i_lca_name: str, hv_lca_name: str):
        super().__init__()
        _set_hidden_attr(self, "_owner_ref", weakref.ref(owner_module))
        self.igam_i_name = igam_i_name
        self.igam_h_name = igam_h_name
        self.i_lca_name  = i_lca_name
        self.hv_lca_name = hv_lca_name

    def forward(self, i_in, hv_in):
        owner = self._owner_ref()
        IGAM_i = getattr(owner, self.igam_i_name)
        IGAM_h = getattr(owner, self.igam_h_name)
        I_LCA  = getattr(owner, self.i_lca_name)
        HV_LCA = getattr(owner, self.hv_lca_name)

        h1 = IGAM_h(hv_in)   # == hv_3 = IGAM_h_up1(hv_3)
        i1 = IGAM_i(i_in)    # == i_dec3 = IGAM_i_up1(i_dec3)

        i2 = I_LCA(i1, h1)   # i_dec2
        h2 = HV_LCA(h1, i1)  # hv_2

        return i2, h2, i1, h1
