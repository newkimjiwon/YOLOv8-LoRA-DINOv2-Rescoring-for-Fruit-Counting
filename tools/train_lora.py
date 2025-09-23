# train_lora.py
"""
단독 YOLOv8 + LoRA 학습 스크립트
- Ultralytics 학습 루틴을 그대로 사용하되, base 가중치는 freeze, LoRA 파라미터만 학습
- 학습 완료 후 (--merge 1) 선택적으로 LoRA를 base에 머지한 체크포인트 저장
"""
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
from yolov8_lora_module import inject_lora_for_yolov8, freeze_base_but_lora, merge_lora_into_base

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data.yaml (train/val/test 포함)")
    ap.add_argument("--model", type=str, default="yolov8s.pt", help="base 모델 가중치")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", type=str, default="train_lora", help="runs/detect/<name>")
    # LoRA 하이퍼
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--only_1x1", type=int, default=1, help="1x1 conv에만 주입")
    ap.add_argument("--min_out_channels", type=int, default=64)
    # merge/export
    ap.add_argument("--merge", type=int, default=0, help="학습 후 LoRA를 base로 머지한 체크포인트 추가 저장")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 모델 로드
    yolo = YOLO(args.model)               # YOLO 객체
    net = yolo.model                      # nn.Module
    net.to(device)

    # 2) LoRA 주입
    net, nrep = inject_lora_for_yolov8(
        net, r=args.lora_r, alpha=args.lora_alpha,
        dropout=args.lora_dropout, only_1x1=bool(args.only_1x1),
        min_out_channels=args.min_out_channels
    )
    print(f"[LoRA] injected into {nrep} Conv2d layers")

    # 3) base freeze, LoRA만 학습
    freeze_base_but_lora(net)

    # 4) 학습 (Ultralytics 내부 옵티마이저는 requires_grad=True 파라미터만 학습)
    yolo.model = net
    yolo.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name
    )

    run_dir = Path("runs/detect") / args.name
    best_pt = run_dir / "weights" / "best.pt"
    print("Best weight:", best_pt)

    # 5) (옵션) LoRA → base merge 체크포인트 저장
    if args.merge:
        print("[LoRA] merging into base Conv2d and saving merged checkpoint...")
        merged = YOLO(best_pt)      # best 체크포인트 불러오기
        merge_lora_into_base(merged.model)
        # 저장: merged 가중치 파일명
        merged_path = run_dir / "weights" / "best_merged.pt"
        merged.save(str(merged_path))
        print("Saved merged:", merged_path)

if __name__ == "__main__":
    main()
