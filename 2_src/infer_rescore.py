# ../2_src/infer_rescore.py

"""
infer_code

python 2_src/infer_rescore.py \
  --img_dir 1_data/Fruits-detection/valid/images \
  --yolo_weights 4_runs/detect/y8s_baseline_exp12/weights/best.pt \
  --mlp_weights 3_weights/custom/rescore_mlp_head.pt

"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

# train_rescore_mlp.py에 정의했던 MLP 클래스를 그대로 가져옵니다.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + DINOv2 Rescoring 추론 스크립트")
    parser.add_argument("--img_dir", type=str, required=True, help="추론할 이미지가 있는 폴더 경로")
    parser.add_argument("--yolo_weights", type=str, required=True, help="원본 탐지를 위한 YOLOv8 가중치")
    parser.add_argument("--mlp_weights", type=str, required=True, help="학습된 MLP 헤드 가중치")
    parser.add_argument("--save_dir", type=str, default="4_runs/rescore_results", help="결과 이미지 저장 폴더")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="최종 결과에 표시할 최소 신뢰도")
    parser.add_argument("--alpha", type=float, default=0.5, help="점수 혼합 비율: new_score = (1-a)*yolo + a*mlp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1. 모든 모델 로드
    print("1. 모델 로드 중...")
    yolo = YOLO(args.yolo_weights)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(device).eval()
    
    # MLP 헤드 로드
    mlp_ckpt = torch.load(args.mlp_weights, map_location=device)
    dino_dim = mlp_ckpt['dino_dim']
    head = MLP(input_dim=dino_dim).to(device)
    head.load_state_dict(mlp_ckpt['state_dict'])
    head.eval()

    dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 2. 결과 저장 폴더 생성
    save_dir_base = Path(args.save_dir)
    save_dir_yolo = save_dir_base / "yolo_only"
    save_dir_rescore = save_dir_base / "with_rescore"
    save_dir_yolo.mkdir(parents=True, exist_ok=True)
    save_dir_rescore.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in Path(args.img_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])

    # 3. 각 이미지에 대해 추론 및 Rescoring 실행
    print("\n2. 추론 및 Rescoring 시작...")
    for img_path in tqdm(image_paths, desc="추론 진행 중"):
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) # 시각화용 BGR 이미지

        # (A) YOLO 단독 추론 및 결과 시각화
        results = yolo.predict(im, conf=args.conf_thresh, verbose=False)[0]
        vis_yolo = img_bgr.copy()
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                label = f"{yolo.names[int(box.cls)]} {conf:.2f}"
                cv2.rectangle(vis_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_yolo, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(save_dir_yolo / img_path.name), vis_yolo)

        # (B) DINOv2 Rescoring 적용 및 결과 시각화
        vis_rescore = img_bgr.copy()
        
        # Rescoring을 위해 conf를 낮춰 더 많은 후보를 고려
        candidates = yolo.predict(im, conf=0.1, verbose=False)[0]
        if candidates.boxes is None:
            cv2.imwrite(str(save_dir_rescore / img_path.name), vis_rescore)
            continue

        for box in candidates.boxes:
            xyxy = box.xyxy[0].tolist()
            
            # 후보 영역을 잘라내어 DINO+MLP로 점수 계산
            crop_im = im.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
            crop_tensor = dino_transform(crop_im).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = dino(crop_tensor)
                mlp_score = torch.sigmoid(head(features)).item() # 0~1 사이의 점수로 변환
            
            yolo_score = box.conf.item()
            
            # YOLO 점수와 MLP 점수를 가중 평균하여 새로운 신뢰도 계산
            new_score = (1 - args.alpha) * yolo_score + args.alpha * mlp_score
            
            if new_score >= args.conf_thresh:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{yolo.names[int(box.cls)]} {new_score:.2f}"
                cv2.rectangle(vis_rescore, (x1, y1), (x2, y2), (255, 165, 0), 2) # 다른 색으로 표시
                cv2.putText(vis_rescore, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.imwrite(str(save_dir_rescore / img_path.name), vis_rescore)

    print(f"\n3. 추론 완료! 결과가 다음 폴더에 저장되었습니다:\n- YOLO 원본: {save_dir_yolo}\n- Rescore 적용: {save_dir_rescore}")

if __name__ == "__main__":
    main()