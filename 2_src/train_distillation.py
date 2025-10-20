# ../2_src/train_distillation.py

"""
train_code

python 2_src/train_distillation.py

"""

import sys
from pathlib import Path
# 이 파일이 있는 폴더(2_src)의 경로를 파이썬의 모듈 검색 경로에 추가합니다.
sys.path.append(str(Path(__file__).parent))
# --- 수정 끝 ---

from distillation_trainer import DistillationTrainer

def main():
    # 1. 학습에 필요한 모든 설정을 딕셔너리로 정의합니다.
    #    data 경로는 이전과 같이 절대 경로를 사용합니다.
    args = {
        'model': '3_weights/base/yolov8s.pt',
        'data': '/mnt/d/Project/Dolo/1_data/Fruits-detection/data.yaml',
        'epochs': 50,
        'batch': 16,
        'imgsz': 640,
        'project': '4_runs/detect',
        'name': 'y8s_distilled_exp1' # 새로운 이름으로 저장
    }

    # 2. YOLO 객체를 거치지 않고, 우리의 커스텀 트레이너를 직접 생성합니다.
    trainer = DistillationTrainer(overrides=args)
    
    # 3. 트레이너의 학습 함수를 직접 호출합니다.
    #    이렇게 하면 우리의 설정이 덮어써질 위험이 없습니다.
    trainer.train()

if __name__ == "__main__":
    main()