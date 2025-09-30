# Lightweight Fruit Detection and Counting with YOLOv8 + LoRA + DINOv2

This project aims at **fruit detection and counting in industrial environments**.  
We applied a **LoRA adapter** to YOLOv8 to build a **lightweight model that can be quickly trained even with limited data**,  
and additionally introduced **DINOv2-based feature extraction + MLP rescoring** to improve detection performance for **small fruits and overlapping fruits**.  
Finally, our goal is to deploy the model in **real-time on Jetson Orin Nano with a webcam**.  

## Project Background

### Limitations of YOLOv8
The original YOLOv8 model achieves fast and accurate results, but its detection performance degrades for **overlapping fruits** or **small-sized fruits**.  
In real industrial scenarios, fruits are often stacked or clustered, making this limitation a major obstacle for practical applications.  

### Proposed Solutions
- **DINOv2 Feature Assistance**  
  Instead of directly using YOLOv8 predictions, we leverage **visual features extracted from DINOv2** and perform **MLP-based rescoring**, enabling better separation of overlapping fruits.  

- **Lightweight Training with LoRA**  
  Since the available fruit dataset is relatively small, we applied **LoRA adapters** to update only a subset of parameters. This improves **data efficiency and training optimization** while reducing computational cost.  

## Results

### YOLOv8 + DINOv2 + LoRA
![val_batch0_labels](https://github.com/user-attachments/assets/34db0a0d-56f6-413a-ab78-3e43cfe4e417)

### YOLOv8 + LoRA
![val_batch0_pred](https://github.com/user-attachments/assets/4c223ad4-eed4-4e75-8562-c840594e4b41)

---

## Key Features
- **YOLOv8 + LoRA**  
  - Update only LoRA adapters instead of the entire parameter set → reduced GPU memory usage and lower data requirements  

- **DINOv2 Teacher + MLP Rescoring**  
  - Refine YOLOv8 predictions using DINOv2 features, improving detection of small and overlapping fruits  

- **Soft-NMS & Tiled Inference**  
  - Maintain high recall even in scenes with many overlapping fruits  

- **Lightweight Deployment Ready**  
  - Exportable to ONNX / TensorRT for real-time inference on Jetson Orin Nano and other edge devices  

## Dataset
This project uses the **Fruit Detection dataset from Kaggle**.  

Download: [Fruit Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection)


## Specifications
- **GPU**: RTX 3090 (24GB)  
- **System**: Ubuntu 22.04.5  
- **PyTorch**: 2.1.0  
