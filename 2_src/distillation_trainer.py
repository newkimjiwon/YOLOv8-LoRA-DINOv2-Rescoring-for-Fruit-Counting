# ../2_src/distillation_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer

class DistillationTrainer(DetectionTrainer):
    """
    YOLOv8에 지식 증류를 적용하기 위한 커스텀 트레이너.
    선생님 모델(DINOv2)의 특징을 학생 모델(YOLOv8)이 모방하도록 학습합니다.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. 선생님 모델(DINOv2) 로드 및 동결
        print("지식 증류: 선생님 모델(DINOv2) 로드 중...")
        # device를 명시적으로 cpu로 설정하고, 나중에 _setup_train에서 올바른 장치로 보냅니다.
        self.teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).cpu().eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print("지식 증류: 선생님 모델 로드 및 동결 완료.")

        # 2. 손실 함수 및 가중치 정의
        self.distillation_loss_fn = nn.MSELoss()
        self.lambda_distill = 10.0
        
        # 3. 특징 추출을 위한 변수 초기화
        self.student_features = None
        self.teacher_features = None
        self.projection = None # 프로젝션 레이어는 모델이 생성된 후 초기화합니다.

    def _setup_train(self, world_size=1):
        """
        모델이 완전히 로드된 후, 학습 시작 직전에 호출되는 메서드.
        Hook을 등록하기에 가장 안전한 시점입니다.
        """
        # 1. 부모 클래스의 _setup_train을 먼저 실행하여 모델을 포함한 모든 것을 준비합니다.
        super()._setup_train(world_size)
        
        # 이제 self.device가 설정되었으므로 선생님 모델을 올바른 장치로 보냅니다.
        self.teacher.to(self.device)
        
        # 2. 이제 self.model이 실제 PyTorch 모델이므로, 안전하게 Hook을 등록할 수 있습니다.
        print("지식 증류: 학생 모델에 Hook 등록 중...")
        student_out_channels = self.model.model[15].cv2.conv.in_channels
        teacher_out_channels = self.teacher.embed_dim
        
        self.projection = nn.Conv2d(student_out_channels, teacher_out_channels, kernel_size=1).to(self.device)
        
        # 학생(YOLOv8) 모델의 중간 레이어(15번)에 Hook을 등록
        self.model.model[15].register_forward_hook(self.get_student_features_hook())
        print("지식 증류: Hook 등록 완료.")

    def get_student_features_hook(self):
        def hook(model, input, output):
            self.student_features = output
        return hook

    def get_loss(self, batch, preds, i):
        # 1. 기존 YOLOv8 손실 계산
        yolo_loss, loss_items = super().get_loss(batch, preds, i)
        
        # 2. 선생님 모델로부터 특징 추출
        with torch.no_grad():
            resized_imgs = F.interpolate(batch["img"], size=(224, 224), mode='bilinear', align_corners=False)
            self.teacher_features = self.teacher.forward_features(resized_imgs)['x_norm_patchtokens']
            B, N, C = self.teacher_features.shape
            H = W = int(N ** 0.5)
            self.teacher_features = self.teacher_features.permute(0, 2, 1).reshape(B, C, H, W)

        # 3. 학생 특징에 프로젝션 적용 및 크기 조절
        projected_student_features = self.projection(self.student_features)
        resized_student_features = F.interpolate(projected_student_features, size=self.teacher_features.shape[2:], mode='bilinear')

        # 4. 지식 증류 손실 계산
        distillation_loss = self.distillation_loss_fn(resized_student_features, self.teacher_features)

        # 5. 최종 손실
        total_loss = yolo_loss + self.lambda_distill * distillation_loss
        
        # 로그에 증류 손실 추가
        # loss_items의 device와 distillation_loss의 device를 일치시켜야 합니다.
        distill_loss_item = distillation_loss.unsqueeze(0).detach().to(loss_items.device)
        loss_items = torch.cat((loss_items, distill_loss_item))
        
        return total_loss, loss_items