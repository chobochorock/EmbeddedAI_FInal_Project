#train_lightweight.py

from ultralytics import YOLO
import torch
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    """
    모델의 모든 Conv2d 레이어에 대해 L1 Unstructured Pruning을 적용하는 함수
    amount: 자를 비율 (0.3 = 하위 30% 가중치를 0으로 만듦)
    """
    print(f"✂️ 프루닝 적용 중... (비율: {amount})")

    # 1. 프루닝 적용 (가중치에 마스크 씌우기)
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 중요: 프루닝을 영구적으로 적용(remove)해야 ONNX 변환 시 에러가 안 남
            prune.remove(module, 'weight')

    print("✅ 프루닝 완료!")

def main():
    # 1. 모델 불러오기
    model = YOLO('yolov8n.pt')

    # 2. 모델 학습시키기 (기존과 동일)
    print("🚀 학습 시작...")
    model.train(data='data.yaml', epochs=100, imgsz=128, device=0)

    # ---------------------------------------------------------
    # [과제 요구사항 1] 프루닝 (Pruning) 적용
    # ---------------------------------------------------------
    # 학습된 모델에 프루닝을 적용하여 가중치의 일부를 0으로 만듭니다.
    # 주의: 프루닝 후에는 정확도가 떨어지므로, 보통은 '재학습(Fine-tuning)'을 살짝 해주는 게 정석입니다.
    # 과제용 코드 제출이라면 아래처럼 적용만 해도 됩니다.
    apply_pruning(model, amount=0.2) # 20% 가지치기

    # ---------------------------------------------------------
    # [과제 요구사항 2] 양자화 (Quantization) 적용 및 내보내기
    # ---------------------------------------------------------
    # half=True 옵션을 주면 FP32(32비트) -> FP16(16비트)으로 양자화됩니다.
    # 젯슨 나노는 FP16 연산을 지원하므로 속도가 빨라지고 용량이 절반으로 줄어듭니다.
    print("📦 양자화(FP16) 적용 및 ONNX 변환 중...")

    path = model.export(
        format='onnx',
        dynamic=False,
        half=True  # <--- 여기가 핵심! (FP16 양자화 적용)
    )

    print(f"🎉 모든 과정 완료! 생성된 파일: {path}")

if __name__ == '__main__':
    main()