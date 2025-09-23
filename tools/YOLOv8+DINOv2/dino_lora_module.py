# ~/DOLO/tools/dino_lora_module.py
import math, torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        if r > 0:
            self.A = nn.Parameter(torch.zeros((r, base.in_features)))
            self.B = nn.Parameter(torch.zeros((base.out_features, r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = alpha / r
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.base(x)
        if self.r and self.r > 0:
            lora = (x @ self.A.t()) @ self.B.t()
            out = out + self.dropout(lora) * self.scaling
        return out

def _set_module(parent, name, new_module):
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

def inject_lora_for_dinov2(vit: nn.Module, r=8, alpha=16, dropout=0.05):
    """
    Dinov2 ViT는 보통 blocks[i].attn.qkv / blocks[i].attn.proj 등의 Linear 사용.
    해당 Linear에 LoRA 주입.
    """
    target_suffix = ("attn.qkv", "attn.proj")
    replaced = 0
    for name, module in vit.named_modules():
        if isinstance(module, nn.Linear):
            # 이름 끝이 qkv/proj 인지 확인 (dinov2 구조 기준)
            if any(name.endswith(suf) for suf in target_suffix):
                _set_module(vit, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return vit, replaced

def lora_parameters(module: nn.Module):
    for n, p in module.named_parameters():
        if any(k in n for k in (".A", ".B")):
            yield p
