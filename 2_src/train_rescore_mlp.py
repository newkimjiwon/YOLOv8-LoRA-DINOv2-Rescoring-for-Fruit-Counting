# ../2_src/train_rescore_mlp.py

"""
train_code

python 2_src/train_rescore_mlp.py \
  --data_root 1_data/Fruits-detection \
  --yolo_weights 4_runs/detect/y8s_baseline_exp12/weights/best.pt

"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

# --- 유틸리티 함수 ---
def iou(boxA, boxB):
    """ 두 바운딩 박스 간의 IoU(Intersection over Union) 계산 """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_ground_truths(lbl_path, W, H):
    """ YOLO 형식의 txt 라벨 파일을 [x1, y1, x2, y2] 형식으로 변환 """
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W
                y2 = (cy + h/2) * H
                boxes.append([x1, y1, x2, y2])
    return boxes

# --- MLP 모델 정의 ---
class MLP(nn.Module):
    """ DINOv2 특징을 입력받아 TP/FP 확률을 출력하는 간단한 MLP """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- 메인 학습 함수 ---
def main():
    parser = argparse.ArgumentParser(description="DINOv2+MLP Rescoring Head 학습 스크립트")
    parser.add_argument("--data_root", type=str, required=True, help="데이터셋 루트 폴더 경로 (train/, valid/ 포함)")
    parser.add_argument("--yolo_weights", type=str, required=True, help="후보 탐지를 위한 학습된 YOLOv8 가중치 경로")
    parser.add_argument("--save_dir", type=str, default="3_weights/custom", help="학습된 MLP 헤드 저장 폴더")
    parser.add_argument("--epochs", type=int, default=5, help="MLP 학습 에포크")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--batch_size", type=int, default=128, help="배치 사이즈")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1. 모델 로드 (YOLOv8, DINOv2)
    print("1. YOLOv8 및 DINOv2 모델 로드 중...")
    yolo = YOLO(args.yolo_weights)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    
    # DINOv2 특징 추출을 위한 이미지 변환 파이프라인
    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 2. 학습 데이터 수집 (YOLO 후보 -> TP/FP 라벨링)
    print("\n2. 학습 데이터 수집 시작 (YOLO 후보 -> TP/FP 라벨링)...")
    X_crops = []  # 잘라낸 이미지 텐서
    Y_labels = [] # 1 (TP) 또는 0 (FP) 라벨

    # 검증 데이터셋을 사용하여 MLP 학습 데이터를 생성
    img_dir = Path(args.data_root) / "valid" / "images"
    lbl_dir = Path(args.data_root) / "valid" / "labels"
    image_paths = sorted(list(img_dir.glob("*.*")))

    for img_path in tqdm(image_paths, desc="데이터 수집 중"):
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        
        # 정답 바운딩 박스 로드
        gt_boxes = get_ground_truths(lbl_dir / f"{img_path.stem}.txt", W, H)
        
        # YOLO로 후보 박스 예측 (conf를 낮춰 많은 후보 확보)
        results = yolo.predict(im, conf=0.1, verbose=False)
        
        if results[0].boxes is None:
            continue

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            # 후보 박스가 정답(TP)인지 아닌지 판단
            is_tp = any(iou(xyxy, gt_box) > 0.5 for gt_box in gt_boxes)
            
            # 이미지 자르기 및 변환
            crop = im.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
            crop_tensor = dino_transform(crop)
            
            X_crops.append(crop_tensor)
            Y_labels.append(1.0 if is_tp else 0.0)

    if not X_crops:
        print("수집된 후보가 없습니다. YOLO --conf 값을 확인하거나 경로를 확인하세요.")
        return
        
    X = torch.stack(X_crops).to(device)
    Y = torch.tensor(Y_labels, dtype=torch.float32, device=device).unsqueeze(1)
    print(f"\n총 {len(X)}개의 학습 샘플 수집 완료 (TP: {int(Y.sum().item())}, FP: {len(Y) - int(Y.sum().item())})")

    # 3. MLP 헤드 학습
    print("\n3. MLP 헤드 학습 시작...")
    with torch.no_grad():
        dino_dim = dino(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    
    head = MLP(input_dim=dino_dim).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss() # 이진 분류에 적합한 손실 함수

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for xb, yb in tqdm(loader, desc=f"에포크 {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            with torch.no_grad():
                features = dino(xb) # DINOv2로 특징 추출
            logits = head(features) # MLP로 TP/FP 예측
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"에포크 {epoch} 완료, 평균 손실: {total_loss / len(loader):.4f}")

    # 4. 학습된 MLP 헤드 저장
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "rescore_mlp_head.pt"
    
    checkpoint = {
        'state_dict': head.state_dict(),
        'dino_dim': dino_dim,
    }
    torch.save(checkpoint, model_path)
    print(f"\n4. 학습된 MLP 헤드를 여기에 저장했습니다: {model_path}")

if __name__ == "__main__":
    main()