import sys
import cv2
import numpy as np
import torch
import jetson.utils
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
MODEL_PATH = "best.pt"         # YOLOv8 학습된 모델 (PC에서 가져온 것)
FONT_PATH  = "NotoSansKR-Regular.ttf" # 폰트 경로
CAMERA_DEVICE = "csi://0"
# ---------------------------------------------------------

def main():
    print(f"🚀 모델 로딩 중: {MODEL_PATH} ...")
    # YOLO 모델 로드 (자동으로 TensorRT 엔진이 있으면 그걸 씁니다)
    model = YOLO(MODEL_PATH)

    # 카메라 및 디스플레이 준비
    camera = jetson.utils.videoSource(CAMERA_DEVICE)
    display = jetson.utils.videoOutput("display://0")
    
    # 폰트 로드 (PIL 사용)
    try:
        font = ImageFont.truetype(FONT_PATH, 30)
    except:
        print("폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    print("🎥 영상 감지 시작! (Ctrl+C로 종료)")

    while display.IsStreaming():
        # 1. 이미지 캡처 (CUDA 메모리)
        img_cuda = camera.Capture()
        if img_cuda is None: continue

        # 2. YOLO 입력을 위해 Numpy 변환 (CUDA -> CPU)
        # jetson.utils는 RGBA float32 형식을 줌 -> uint8 변환 필요
        img_numpy = jetson.utils.cudaToNumpy(img_cuda)
        img_rgb = np.array(img_numpy, dtype=np.uint8)

        # 3. 객체 검출 (YOLOv8)
        # verbose=False: 터미널에 로그 너무 많이 뜨는 것 방지
        results = model(img_rgb, verbose=False)

        # 4. 결과 그리기 (PIL 사용)
        pil_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_image)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 좌표 및 정보 추출
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # 클래스 이름 (한자)
                class_name = model.names[cls]
                label = f"{class_name} {conf*100:.1f}%"

                # 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                # 글씨 그리기
                draw.text((x1, y1 - 35), label, font=font, fill=(255, 255, 255))

        # 5. 화면 출력
        # PIL 이미지를 다시 Numpy(BGR)로 변환하여 OpenCV 창에 표시
        # (jetson.utils.videoOutput을 쓰려면 다시 CUDA로 올려야 해서 복잡함)
        final_img = np.array(pil_image)
        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Hanja Detection", final_img_bgr)
        
        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()