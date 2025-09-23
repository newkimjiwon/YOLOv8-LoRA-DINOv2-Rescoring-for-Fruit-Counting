# infer_lora.py
"""
YOLOv8(+LoRA) 추론 스크립트 (Soft-NMS + 타일 추론 옵션 추가)
"""

import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import os

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="best.pt 또는 best_merged.pt")
    ap.add_argument("--source", type=str, required=True, help="이미지/폴더/비디오/웹캠(0)")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--save_dir", type=str, default="runs/predict_lora")
    ap.add_argument("--soft_nms", action="store_true", help="Soft-NMS 활성화")
    ap.add_argument("--tile", type=int, default=0, help="타일 추론 (예: 2면 2x2 타일)")
    return ap.parse_args()

# -------------------
# Soft-NMS 구현 (간단 버전)
# -------------------
def soft_nms(boxes, scores, iou_thresh=0.7, sigma=0.5, Nt=0.3, method=2):
    """
    boxes: [N,4], scores: [N]
    method=2 → Gaussian Soft-NMS
    """
    N = len(boxes)
    for i in range(N):
        maxscore = scores[i]
        maxpos = i

        tx1, ty1, tx2, ty2 = boxes[i]

        pos = i + 1
        while pos < N:
            x1, y1, x2, y2 = boxes[pos]

            xx1 = max(tx1, x1)
            yy1 = max(ty1, y1)
            xx2 = min(tx2, x2)
            yy2 = min(ty2, y2)

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h

            area = (tx2 - tx1 + 1) * (ty2 - ty1 + 1)
            area_pos = (x2 - x1 + 1) * (y2 - y1 + 1)
            ovr = inter / (area + area_pos - inter)

            if method == 2:  # Gaussian
                weight = np.exp(-(ovr * ovr) / sigma)
            else:  # linear
                if ovr > Nt:
                    weight = 1 - ovr
                else:
                    weight = 1
            scores[pos] = weight * scores[pos]

            if scores[pos] < 1e-3:
                boxes = np.delete(boxes, pos, axis=0)
                scores = np.delete(scores, pos, axis=0)
                N = len(boxes)
            else:
                pos += 1
    return boxes, scores

# -------------------
# 타일 추론
# -------------------
def run_tiled_inference(model, img_path, tile=2, imgsz=896, conf=0.25, iou=0.7):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    tile_h, tile_w = H // tile, W // tile

    results_all = []
    for r in range(tile):
        for c in range(tile):
            x1, y1 = c * tile_w, r * tile_h
            x2, y2 = (c+1) * tile_w, (r+1) * tile_h
            crop = img[y1:y2, x1:x2]

            res = model.predict(
                source=crop,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False
            )
            for box in res[0].boxes.xyxy.cpu().numpy():
                bx = [box[0]+x1, box[1]+y1, box[2]+x1, box[3]+y1]
                results_all.append(bx)
    return results_all

# -------------------
# 메인 실행
# -------------------
def main():
    args = parse_args()
    model = YOLO(args.weights)

    os.makedirs(f"{args.save_dir}/pred", exist_ok=True)

    if args.tile > 0:
        print(f"[Tile inference] {args.tile}x{args.tile} 적용 중...")
        results = run_tiled_inference(model, args.source, tile=args.tile,
                                      imgsz=args.imgsz, conf=args.conf, iou=args.iou)
        print("타일 추론 결과 box 수:", len(results))
    else:
        model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            project=args.save_dir,
            name="pred",
            save=True
        )
        print("Saved to:", f"{args.save_dir}/pred")

if __name__ == "__main__":
    main()
