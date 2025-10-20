# ../2_src/module_yolo_lora.py
import math
from typing import Tuple, Iterable
import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    """
    nn.Conv2d 레이어를 위한 LoRA 구현.
    y = h(x) + (alpha / r) * B(A(x))
    A: 채널 차원을 r로 줄이는 1x1 Conv
    B: 채널 차원을 다시 원래대로 복원하는 1x1 Conv
    """
    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.base = base_conv
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if self.r > 0:
            # 저랭크(low-rank) 행렬 A와 B를 1x1 Conv로 구현
            self.A = nn.Conv2d(self.base.in_channels, self.r, kernel_size=1, bias=False)
            self.B = nn.Conv2d(self.r, self.base.out_channels, kernel_size=1, bias=False)
            
            # 가중치 초기화
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None
            self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            # LoRA 경로 추가
            y = y + self.dropout(self.B(self.A(x))) * self.scaling
        return y

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        """LoRA 파라미터(A, B)만 반환하는 제너레이터"""
        if self.r > 0:
            yield from self.A.parameters()
            yield from self.B.parameters()

def _set_module(root: nn.Module, dotted_name: str, new_mod: nn.Module):
    """'model.layer1.conv' 같은 이름으로 모듈을 찾아 교체"""
    obj = root
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_mod)

def inject_lora_to_yolo(model: nn.Module, r: int, alpha: int, dropout: float):
    """YOLOv8 모델을 순회하며 Conv2d를 LoRAConv2d로 교체"""
    replaced_count = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1) and m.groups == 1:
            lora_layer = LoRAConv2d(m, r=r, alpha=alpha, dropout=dropout)
            _set_module(model, name, lora_layer)
            replaced_count += 1
    print(f"[LoRA] {replaced_count}개의 Conv2d 레이어를 LoRAConv2d로 교체했습니다.")
    return model

def freeze_base_but_lora(model: nn.Module):
    """모델의 모든 파라미터를 동결시키고 LoRA 파라미터만 학습 가능하게 설정"""
    for p in model.parameters():
        p.requires_grad = False
    
    # LoRA 파라미터만 requires_grad=True로 설정
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            for p in m.lora_parameters:
                p.requires_grad = True
    print("[LoRA] 기본 가중치를 동결하고 LoRA 파라미터만 학습하도록 설정했습니다.")

@torch.no_grad()
def merge_lora_into_base(model: nn.Module):
    """학습된 LoRA 가중치를 원래 Conv2d 가중치에 합침"""
    merged_count = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, LoRAConv2d) and m.r > 0:
            # W_new = W_base + B @ A
            W_delta = (m.B.weight @ m.A.weight).view_as(m.base.weight)
            m.base.weight.copy_(m.base.weight + W_delta * m.scaling)
            
            # LoRAConv2d를 다시 원래의 nn.Conv2d로 교체
            _set_module(model, name, m.base)
            merged_count += 1
    if merged_count > 0:
        print(f"[LoRA] {merged_count}개의 LoRA 가중치를 기본 Conv2d 레이어에 병합했습니다.")