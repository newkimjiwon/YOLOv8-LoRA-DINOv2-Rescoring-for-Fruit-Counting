# YOLOv8 + LoRA + DINOv2 기반 경량 과일 탐지 및 개수 산출

이 프로젝트는 **공정 환경에서의 과일 탐지 및 개수 산출**을 목표로 합니다.  
YOLOv8에 **LoRA 어댑터**를 적용하여 **적은 데이터로도 빠르게 학습 가능한 경량 모델**을 만들고,  
추가적으로 **DINOv2 기반 특징 추출 + MLP 리스코어링**을 통해 **작은 과일 및 겹치는 과일 탐지 성능**을 향상시켰습니다.  
최종적으로는 **Jetson Orin Nano + 웹캠 환경**에서 실시간 적용을 목표로 합니다.  


## 주요 특징
- **YOLOv8 + LoRA**  
  - 전체 파라미터 학습 대신 LoRA만 업데이트 → GPU 메모리/학습 데이터 요구량 감소  
- **DINOv2 Teacher + MLP Rescoring**  
  - YOLOv8 결과를 DINOv3 특징으로 보정하여 작은 과일, 겹친 과일 탐지 성능 향상  
- **Soft-NMS & 타일 추론 지원**  
  - 겹치는 과일이 많은 장면에서도 높은 리콜율 확보  
- **경량 배포 준비**  
  - ONNX / TensorRT 변환을 통해 Jetson Orin Nano 등 엣지 디바이스에서 실시간 동작 가능  


## 프로젝트 구조
project/
├── tools/ # 학습 및 추론 스크립트 <br/>
│ ├── train_lora.py # YOLOv8 + LoRA 학습 <br/>
│ ├── infer_lora.py # YOLOv8 + LoRA 추론 (Soft-NMS/타일 옵션 포함) <br/>
│ ├── yolov8_lora_module.py# LoRA 모듈 정의 <br/>
│ ├── rescore_train.py # DINO + MLP 리스코어링 학습 <br/>
│ ├── rescore_infer.py # DINO + MLP 리스코어링 추론 <br/>
│ └── dino_lora_module.py # DINO + LoRA 모듈 정의 <br/>
├── runs/ # 학습 결과 (weights, 로그, 결과 이미지) <br/>
└── fruit_detection_raw/ # 데이터셋 (깃허브에는 미포함) <br/>


## 데이터셋
본 프로젝트에서는 Kaggle의 Fruit Detection 데이터셋을 사용했습니다.  

다운로드: [Fruit Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection)

데이터셋 구조 예시:
fruit_detection_raw/ <br/>
└── Fruits-detection/ <br/>
├── train/ <br/>
│ ├── images/ <br/>
│ └── labels/ <br/>
└── valid/ <br/>
├── images/ <br/>
└── labels/ <br/>


## 사용 방법


### 1) 학습 (YOLOv8 + LoRA)
```bash
python tools/train_lora.py \
  --data fruit_detection_raw/Fruits-detection/data.yaml \
  --model yolov8s.pt \
  --imgsz 1280 --epochs 60 --batch 16 \
  --name y8s_lora_1280_r8

