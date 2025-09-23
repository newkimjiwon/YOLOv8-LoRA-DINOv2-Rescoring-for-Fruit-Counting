# ~/DOLO/tools/rescore_train.py
import os, argparse, numpy as np
from pathlib import Path
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from ultralytics import YOLO

from tools.dino_lora_module import inject_lora_for_dinov2, lora_parameters

# ---------------- Utils ----------------
def yolo_txt_to_xyxy(lbl_path: Path, W: int, H: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) >= 5:
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
            x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
            boxes.append((c, [x1,y1,x2,y2]))
    return boxes

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2,bx2) - max(ax1,bx1)) * max(0, min(ay2,by2) - max(ay1,by1))
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(ua, 1e-6)

# ---------------- DINO loaders ----------------
def load_dino(backbone: str, device: torch.device, v3_repo: str=None, v3_weights: str=None):
    """
    backbone: 'dino_v2' (default) | 'dino_v3'
    dino_v2: torch.hub로 즉시 사용 가능
    dino_v3: 사용자가 repo/weights를 제공해야 함 (플러그인 방식)
    """
    if backbone == "dino_v3":
        if not (v3_repo and v3_weights):
            raise SystemExit("DINOv3 사용 시 --v3-repo 와 --v3-weights 경로를 제공하세요.")
        import sys
        sys.path.append(os.path.abspath(v3_repo))
        # 예시) from dinov3 import build_model; model = build_model(v3_weights) 등
        # 실제 구현은 가져온 레포 구조에 맞춰 수정 필요
        raise SystemExit("DINOv3 로더는 레포 구조에 맞게 커스터마이징하세요.")
    else:
        import torch.hub as hub
        dino = hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dino.to(device).eval()
        return dino

# ---------------- MLP Head ----------------
class MLP(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="YOLO data root (images/, labels/ 포함)")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs/rescoring")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.15, help="후보 많이 모으기 위해 낮춤")
    ap.add_argument("--iou", type=float, default=0.6)

    ap.add_argument("--backbone", type=str, default="dino_v2", choices=["dino_v2","dino_v3"])
    ap.add_argument("--v3_repo", type=str, default=None)
    ap.add_argument("--v3_weights", type=str, default=None)

    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--lora_r", type=int, default=0, help=">0이면 LoRA 주입")
    ap.add_argument("--lora_train", type=int, default=0, help="1이면 LoRA 파라미터도 함께 학습")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(os.path.expanduser(args.data_root))
    save_dir = Path(os.path.expanduser(args.save_dir)); save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 모델들
    yolo = YOLO(os.path.expanduser(args.yolo_weights))
    dino = load_dino(args.backbone, device, args.v3_repo, args.v3_weights)

    # (옵션) LoRA 주입
    if args.lora_r > 0 and args.backbone == "dino_v2":
        dino, replaced = inject_lora_for_dinov2(dino, r=args.lora_r, alpha=16, dropout=0.05)
        print(f"[LoRA] injected to {replaced} linear layers (dinov2)")

    # 2) 데이터 수집 (YOLO 후보 -> crop -> TP/FP label)
    t = transforms.Compose([
        transforms.Resize((args.crop, args.crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    X_imgs = []  # crop tensor
    Y = []       # 1(TP)/0(FP)

    images = sorted((data_root / args.split / "images").glob("*.*"))
    for imgp in images:
        im = Image.open(imgp).convert("RGB"); W,H = im.size
        gt = yolo_txt_to_xyxy(data_root / args.split / "labels" / f"{imgp.stem}.txt", W, H)

        r = yolo.predict(str(imgp), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: 
            continue
        for b in r.boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            crop = im.crop((max(0,x1), max(0,y1), min(W,x2), min(H,y2)))
            X_imgs.append(t(crop))
            is_tp = any(iou([x1,y1,x2,y2], g[1]) >= 0.5 for g in gt)
            Y.append(1.0 if is_tp else 0.0)

    if len(X_imgs) == 0:
        raise SystemExit("수집된 후보가 없습니다. --conf를 더 낮추거나 split/경로를 확인하세요.")

    X = torch.stack(X_imgs).to(device)   # [N,3,224,224]
    Y = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(1)

    # 3) 임베딩 차원 파악 및 헤드 준비
    with torch.no_grad():
        zdim = dino(torch.zeros(1,3,args.crop,args.crop, device=device)).shape[-1]
    head = MLP(dim=zdim).to(device)

    # 4) 최적화 대상
    params = list(head.parameters())
    if args.lora_train and args.lora_r > 0:
        params += list(lora_parameters(dino))  # LoRA 파라미터만 학습

    opt = optim.AdamW(params, lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # 5) 학습
    N = X.size(0); bs = max(1, args.bs)
    idx_all = torch.arange(N, device=device)

    for ep in range(1, args.epochs+1):
        perm = idx_all[torch.randperm(N, device=device)]
        total_loss = 0.0; steps = 0
        for i in range(0, N, bs):
            sel = perm[i:i+bs]
            xb, yb = X[sel], Y[sel]
            opt.zero_grad()
            z = dino(xb)          # [B, zdim]
            logits = head(z)      # [B,1]
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()); steps += 1
        print(f"[ep {ep}/{args.epochs}] loss={total_loss/max(1,steps):.4f}")

    # 6) 저장
    ckpt = {
        "head": head.state_dict(),
        "zdim": zdim,
        "crop": args.crop,
        "backbone": args.backbone,
        "lora_r": args.lora_r
    }
    out = save_dir / "rescore_mlp.pt"
    torch.save(ckpt, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
