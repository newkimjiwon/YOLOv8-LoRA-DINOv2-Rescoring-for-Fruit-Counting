# 파일 경로: 2_src/train_yolo.py
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# LoRA 관련 함수들을 import
from module_yolo_lora import inject_lora_to_yolo, freeze_base_but_lora, merge_lora_into_base

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Baseline and LoRA Training Script")
    # --- 기본 학습 인자 ---
    parser.add_argument("--data", type=str, required=True, help="data.yaml 파일 경로")
    parser.add_argument("--model", type=str, required=True, help="사전 학습된 모델 가중치 경로 (e.g., yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="학습 이미지 크기")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")
    parser.add_argument("--batch", type=int, default=16, help="배치 사이즈")
    parser.add_argument("--project", type=str, default="4_runs/detect", help="학습 결과 저장 프로젝트 폴더")
    parser.add_argument("--name", type=str, required=True, help="학습 결과 저장 폴더 이름")

    # --- LoRA 관련 인자 ---
    parser.add_argument("--use_lora", action="store_true", help="이 플래그를 사용하면 LoRA 학습을 활성화합니다.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA의 rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA의 alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA의 dropout")
    
    # --- 학습 후 처리 ---
    parser.add_argument("--merge_on_finish", action="store_true", help="LoRA 학습 완료 후 가중치를 병합하여 저장합니다.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("1. YOLO 모델 로드 중...")
    yolo = YOLO(args.model)
    net = yolo.model  # nn.Module

    if args.use_lora:
        print("\n2. LoRA 학습을 시작합니다.")
        # LoRA 레이어를 모델에 주입
        net = inject_lora_to_yolo(net, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        # LoRA 파라미터만 학습하도록 설정
        freeze_base_but_lora(net)
        yolo.model = net
    else:
        print("\n2. Baseline 학습을 시작합니다. (LoRA 비활성화)")

    print("\n3. 모델 학습...")
    yolo.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name
    )

    print("\n4. 학습 완료.")
    
    # LoRA 학습 후 병합 옵션이 켜져있으면 실행
    if args.use_lora and args.merge_on_finish:
        print("\n5. LoRA 가중치를 기본 모델에 병합합니다.")
        # 학습된 best.pt 경로
        run_dir = Path(args.project) / args.name
        best_pt_path = run_dir / "weights" / "best.pt"
        
        # best.pt를 다시 로드하여 병합 진행
        merged_model = YOLO(best_pt_path)
        merge_lora_into_base(merged_model.model)
        
        # 병합된 모델 저장
        merged_path = run_dir / "weights" / "best_merged.pt"
        merged_model.save(str(merged_path))
        print(f"병합된 모델을 여기에 저장했습니다: {merged_path}")

if __name__ == "__main__":
    main()

"""

Train_Code

실험 1: Baseline YOLOv8 모델 학습

python 2_src/train_yolo.py \
  --data 1_data/Fruits-detection/data.yaml \
  --model 3_weights/base/yolov8s.pt \
  --imgsz 640 \
  --epochs 10 \
  --batch 16 \
  --name y8s_baseline_exp1

10 epochs completed in 0.176 hours


실험 2: YOLOv8 + LoRA 모델 학습

python 2_src/train_yolo.py \
  --data 1_data/Fruits-detection/data.yaml \
  --model 3_weights/base/yolov8s.pt \
  --imgsz 640 \
  --epochs 10 \
  --batch 16 \
  --name y8s_lora_exp1 \
  --use_lora

10 epochs completed in 0.175 hours.

"""