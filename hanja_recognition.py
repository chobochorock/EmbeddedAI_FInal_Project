import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torch2trt import torch2trt
from torchvision import transforms
import jetson.utils # 카메라용 라이브러리

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# YOLO 모델 경로 (TensorRT 엔진(.engine) 추천, 없으면 .pt)
YOLO_PATH = 'best.pt' 

# ResNet 모델 경로 (전체 정보가 담긴 딕셔너리 파일)
RESNET_PATH = 'recognition_model_final.pth'

# 폰트 경로 (한자를 출력하기 위해 필수!)
# 시스템에 있는 한글/한자 폰트 경로로 수정하세요.
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" 

# ==========================================
# 2. 모델 로드 (초기화)
# ==========================================
print("모델을 로딩 중입니다...")

# (1) YOLO 로드
detector = YOLO(YOLO_PATH)

# (2) ResNet 로드
device = torch.device("cuda")
checkpoint = torch.load(RESNET_PATH, map_location=device)

# 저장된 정보 복구
idx2char = checkpoint['idx2char']
classes = checkpoint['classes']

# 모델 구조 정의 (학습 때 사용한 구조와 동일해야 함, 예시는 SimpleCNN)
# 만약 ResNet18을 썼다면 그 함수를 가져와야 합니다.
# from simple_cnn import SimpleCNN # <-- [주의] SimpleCNN 클래스가 있는 파일 import
# classifier = SimpleCNN(num_classes=len(classes)) 
from torchvision import models

def get_resnet_for_50x50(num_classes):
    # 1. ResNet18 불러오기 (Pretrained는 ImageNet(RGB) 기준이라 구조 변경 시 가중치 매칭이 까다로울 수 있어 False 추천)
    model = models.resnet18(weights=None) 
    
    # 2. [수정 포인트] 입력 채널을 3 -> 1로 변경
    # 원래 ResNet의 첫 conv1은 (3, 64, 7x7, stride=2)입니다.
    # 50x50 처럼 작은 이미지는 정보를 너무 많이 잃으므로 (1, 64, 3x3, stride=1)로 바꿉니다.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 3. 첫 번째 Pooling 제거 (선택 사항)
    # 50x50 이미지는 너무 작아지면 안 되므로 maxpool을 건너뛰게 할 수도 있습니다.
    # 여기서는 유지하되, 위에서 stride를 1로 줄여서 정보 손실을 막았습니다.
    
    # 4. 마지막 출력층(FC)을 우리 클래스 개수에 맞게 변경
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

classifier = get_resnet_for_50x50(num_classes=4803) # 4803개의 한자 종류
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.to(device).eval()

# (3) ResNet용 전처리 (학습 때와 동일)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("모델 로딩 완료!")

# ==========================================
# 3. 메인 실행 루프
# ==========================================
def main():
    # 커맨드라인 인자로 카메라/화면 설정 받기
    # 예: python3 realtime_demo.py --input=csi://0 --output=display://0
    # 기본값: csi://0 (RPi 카메라), display://0 (연결된 모니터)
    args = sys.argv
    input_uri = "csi://0" # USB카메라면 "/dev/video0"
    output_uri = "display://0"
    
    # 인자 파싱 (간단 구현)
    for arg in args:
        if "input=" in arg: input_uri = arg.split("=")[1]
        if "output=" in arg: output_uri = arg.split("=")[1]

    # 카메라 및 디스플레이 객체 생성 (jetson.utils 활용)
    camera = jetson.utils.videoSource(input_uri)
    display = jetson.utils.videoOutput(output_uri)
    
    font = ImageFont.truetype(FONT_PATH, 30) # 폰트 크기 30

    while display.IsStreaming():
        # 1. 카메라 프레임 캡처 (CUDA 메모리에 저장됨)
        img_cuda = camera.Capture()
        
        if img_cuda is None:
            continue

        # 2. YOLO 처리를 위해 Numpy로 변환 (CUDA -> CPU Numpy)
        # jetson.utils는 기본적으로 RGBA float32 형식을 씁니다.
        # YOLO는 uint8 RGB/BGR을 좋아하므로 변환이 필요합니다.
        img_numpy = jetson.utils.cudaToNumpy(img_cuda)
        
        # RGBA(float) -> RGB(uint8) 변환
        # jetson.utils 이미지는 0~255 값을 가지지만 float32일 수 있음
        img_rgb = np.array(img_numpy, dtype=np.uint8) 
        # 만약 색상이 이상하면 cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB) 사용
        
        # 3. YOLO 탐지 실행
        results = detector(img_rgb, verbose=False)
        
        # PIL 이미지로 변환 (그리기 용도)
        pil_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 4. 탐지된 박스마다 ResNet 돌리기
        for result in results:
            for box in result.boxes:
                # 좌표 추출
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 이미지 잘라내기 (Crop)
                # 좌표가 이미지 범위를 벗어나지 않게 클램핑
                h, w, _ = img_rgb.shape
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                if x2 - x1 < 5 or y2 - y1 < 5: continue # 너무 작은 박스 무시

                crop = img_rgb[y1:y2, x1:x2]
                
                # ResNet 전처리: PIL변환 -> 흑백 -> 리사이즈 -> 텐서
                crop_pil = Image.fromarray(crop).convert('L').resize((50, 50))
                input_tensor = transform(crop_pil).unsqueeze(0).to(device)
                
                # ResNet 추론
                with torch.no_grad():
                    output = classifier(input_tensor)
                    pred_idx = output.argmax(dim=1).item()
                    char_result = idx2char[pred_idx]
                
                # 5. 결과 그리기 (PIL 사용 - 한자 출력 가능)
                # 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                # 글자 쓰기 (박스 위에)
                draw.text((x1, y1 - 35), char_result, font=font, fill=(255, 0, 0))

        # 6. 결과 출력
        # PIL 이미지를 다시 Numpy로 바꿔서 jetson.utils는 못쓰고,
        # 그냥 cv2.imshow를 쓰거나 다시 cuda로 올려야 하는데,
        # 편의상 cv2.imshow를 쓰거나, 여기서는 display 객체 호환을 위해 변환함.
        
        # 가장 쉬운 방법: 그냥 OpenCV 창 띄우기 (Jetson에서는 이것도 빠름)
        final_img = np.array(pil_image)
        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Handwriting Recognition", final_img_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS 출력
        print(f"FPS: {display.GetFrameRate()}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
