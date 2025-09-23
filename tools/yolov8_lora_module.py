# yolov8_lora_module.py
import math
from typing import Tuple, Iterable
import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    """
    LoRA for Conv2d (channel low-rank, 1x1 경로)
    y = Conv_base(x) + scale * Dropout( Conv_B( Conv_A(x) ) )
    - A: in -> r, 1x1
    - B: r  -> out, 1x1
    """
    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base = base_conv
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            # 1x1 경로(채널 저랭크)
            self.A = nn.Conv2d(self.base.in_channels, self.r, kernel_size=1, bias=False)
            self.B = nn.Conv2d(self.r, self.base.out_channels, kernel_size=1, bias=False)
            # kaiming 초기화
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None
            self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0 and self.A is not None and self.B is not None:
            y = y + self.dropout(self.B(self.A(x))) * self.scaling
        return y

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r > 0 and self.A is not None and self.B is not None:
            yield from self.A.parameters()
            yield from self.B.parameters()

def _set_module(root: nn.Module, dotted_name: str, new_mod: nn.Module):
    obj = root
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_mod)

def inject_lora_for_yolov8(model: nn.Module, r: int = 8, alpha: int = 16,
                           dropout: float = 0.0, only_1x1: bool = True,
                           min_out_channels: int = 64) -> Tuple[nn.Module, int]:
    """
    YOLOv8 nn.Module 에 LoRAConv2d 주입
    - only_1x1=True: kernel=1 conv에만 주입 (안전)
    - min_out_channels: 너무 작은 conv는 제외(효율)
    반환: (모델, 주입개수)
    """
    replaced = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            if m.groups != 1:
                continue  # depthwise 등 제외
            if only_1x1 and (m.kernel_size != (1, 1)):
                continue
            if m.out_channels < min_out_channels:
                continue
            # 감싸기
            lora = LoRAConv2d(m, r=r, alpha=alpha, dropout=dropout)
            _set_module(model, name, lora)
            replaced += 1
    return model, replaced

def iter_lora_modules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            yield m

def freeze_base_but_lora(model: nn.Module):
    # 전체 freeze
    for p in model.parameters():
        p.requires_grad = False
    # LoRA 파라미터만 학습
    for l in iter_lora_modules(model):
        for p in l.lora_parameters:
            p.requires_grad = True

@torch.no_grad()
def merge_lora_into_base(model: nn.Module):
    """
    LoRAConv2d(A,B) 를 base Conv2d.weight/bias 에 흡수하고,
    모듈을 순수 Conv2d로 치환.
    """
    to_merge = []
    for name, m in model.named_modules():
        if isinstance(m, LoRAConv2d) and m.r > 0 and m.A is not None and m.B is not None:
            to_merge.append((name, m))
    for name, m in to_merge:
        base = m.base
        # deltaW = B(1x1) ∘ A(1x1) 를 weight에 folding (feature-space 합성)
        # 1x1 conv의 합성은 weight 매트릭스 곱과 동일
        # [out,in,1,1]
        W_delta = torch.matmul(m.B.weight.view(m.B.out_channels, m.r),
                               m.A.weight.view(m.r, m.A.in_channels)).view_as(base.weight)
        base.weight.copy_(base.weight + W_delta * m.scaling)
        # 치환: LoRAConv2d -> base Conv2d
        _set_module(model, name, base)
