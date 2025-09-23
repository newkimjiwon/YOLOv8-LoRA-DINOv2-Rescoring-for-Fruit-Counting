# ~/DOLO/tools/rescore_infer.py
import os, argparse, numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

from tools.dino_lora_module import inject_lora_for_dinov2

def soft_nms_py(boxes: np.ndarray, scores: np.ndarray, Nt=0.5, sigma=0.5, thresh=0.25):
    N = boxes.shape[0]
    idxs = np.arange(N)
    for i in range(N):
        maxpos = i + np.argmax(scores[i:])
        boxes[i], boxes[maxpos] = boxes[maxpos].copy(), boxes[i].copy()
        scores[i], scores[maxpos] = scores[maxpos], scores[i]
        idxs[i], idxs[maxpos] = idxs[maxpos], idxs[i]
        for j in range(i+1, N):
            xx1 = max(boxes[i,0], boxes[j,0]); yy1 = max(boxes[i,1], boxes[j,1])
            xx2 = min(boxes[i,2], boxes[j,2]); yy2 = min(boxes[i,3], boxes[j,3])
            w = max(0.0, xx2-xx1); h = max(0.0, yy2-yy1)
            inter = w*h
            area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            area_j = (boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])
            ovr = inter / (area_i + area_j - inter + 1e-6)
            if ovr > Nt:
                scores[j] = scores[j] * np.exp(-(ovr**2)/sigma)
    keep = [k for k,s in zip(idxs, scores) if s > thresh]
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--mlp_ckpt", type=str, required=True)
    ap.add_argument("--data_yaml", type=str, required=True, help="클래스 이름 로딩용")
    ap.add_argument("--save_dir", type=str, default="runs/compare")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf_in", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.5, help="new=(1-a)*yolo + a*mlp")
    ap.add_argument("--soft_nms", type=int, default=1)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--backbone", type=str, default="dino_v2", choices=["dino_v2","dino_v3"])
    ap.add_argument("--lora_r", type=int, default=0)
    ap.add_argument("--v3_repo", type=str, default=None)
    ap.add_argument("--v3_weights", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 클래스 이름
    import yaml
    names = yaml.safe_load(open(os.path.expanduser(args.data_yaml), "r", encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [v for _,v in sorted(names.items(), key=lambda x:int(x[0]))]

    yolo = YOLO(os.path.expanduser(args.yolo_weights))

    # DINO 준비
    if args.backbone == "dino_v3":
        raise SystemExit("DINOv3 로더는 rescore_train과 동일하게 커스터마이징이 필요합니다.")
    else:
        import torch.hub as hub
        dino = hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        if args.lora_r > 0:
            dino, rep = inject_lora_for_dinov2(dino, r=args.lora_r, alpha=16, dropout=0.0)

    # MLP 로드
    ckpt = torch.load(os.path.expanduser(args.mlp_ckpt), map_location=device)
    zdim = ckpt["zdim"]; crop = ckpt.get("crop", 224)

    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        def forward(self, x): return self.net(x)
    head = MLP(zdim).to(device)
    head.load_state_dict(ckpt["head"]); head.eval()

    t = transforms.Compose([
        transforms.Resize((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    save_dir = Path(os.path.expanduser(args.save_dir)); save_dir.mkdir(parents=True, exist_ok=True)
    out_dir_y = save_dir / "yolo"; out_dir_r = save_dir / "rescore"
    (out_dir_y / "images").mkdir(parents=True, exist_ok=True)
    (out_dir_r / "images").mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in Path(os.path.expanduser(args.images_dir)).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])

    for imgp in imgs:
        im = Image.open(imgp).convert("RGB"); W,H = im.size
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        # (1) YOLO 원점수
        r = yolo.predict(str(imgp), conf=args.conf_in, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        y_boxes, y_cls, y_scores = [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                y_boxes.append([x1,y1,x2,y2])
                y_cls.append(int(b.cls.item()))
                y_scores.append(float(b.conf.item()))
        y_keep = list(range(len(y_boxes)))
        if args.soft_nms and len(y_boxes) > 0:
            y_keep = soft_nms_py(np.array(y_boxes), np.array(y_scores), Nt=0.5, sigma=0.5, thresh=args.conf_in)
        y_boxes = [y_boxes[i] for i in y_keep]; y_cls = [y_cls[i] for i in y_keep]
        vis = bgr.copy()
        for (x1,y1,x2,y2), c in zip(y_boxes, y_cls):
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(vis, names[c], (int(x1), max(0,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(str(out_dir_y / "images" / imgp.name), vis)

        # (2) 재스코어 (YOLO 후보 + DINO+MLP)
        rs_boxes, rs_cls, rs_scores = [], [], []
        for b in r.boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            crop_im = im.crop((max(0,x1), max(0,y1), min(W,x2), min(H,y2)))
            x = t(crop_im).unsqueeze(0).to(device)
            with torch.no_grad():
                z = dino(x)              # [1, zdim]
                p = torch.sigmoid(head(z)).item()
            new_s = (1.0 - args.alpha) * float(b.conf.item()) + args.alpha * p
            rs_boxes.append([x1,y1,x2,y2]); rs_cls.append(int(b.cls.item())); rs_scores.append(new_s)
        rs_keep = list(range(len(rs_boxes)))
        if args.soft_nms and len(rs_boxes) > 0:
            rs_keep = soft_nms_py(np.array(rs_boxes), np.array(rs_scores), Nt=0.5, sigma=0.5, thresh=args.conf_in)
        rs_boxes = [rs_boxes[i] for i in rs_keep]; rs_cls = [rs_cls[i] for i in rs_keep]
        vis2 = bgr.copy()
        for (x1,y1,x2,y2), c in zip(rs_boxes, rs_cls):
            cv2.rectangle(vis2, (int(x1),int(y1)), (int(x2),int(y2)), (0,200,255), 2)
            cv2.putText(vis2, names[c], (int(x1), max(0,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.imwrite(str(out_dir_r / "images" / imgp.name), vis2)

    print("Saved:", str(save_dir))

if __name__ == "__main__":
    main()
