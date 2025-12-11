import jetson.inference
import jetson.utils
import sys

# ---------------------------------------------------------
# [설정] 파일 경로를 본인 환경에 맞게 수정하세요.
# ---------------------------------------------------------
MODEL_PATH = "best.onnx"       # 작성자님이 가져온 모델 (best_fp16.onnx라면 이름 변경)
LABEL_PATH = "labels.txt"      # 4803개 한자 순서가 적힌 파일
FONT_PATH  = "NotoSansKR-Regular.ttf" # 한자 지원 폰트 (같은 폴더에 있어야 함)

# 카메라 설정 (CSI 카메라는 "csi://0", USB 웹캠은 "/dev/video0")
CAMERA_DEVICE = "csi://0" 
# ---------------------------------------------------------

def main():
    # 1. 네트워크 로드 (YOLOv8 설정)
    # YOLOv8은 보통 input-blob='images', output-blob='output0' 입니다.
    print(f"🚀 모델 로딩 중: {MODEL_PATH} ...")
    
    net = jetson.inference.detectNet(argv=[
        f'--model={MODEL_PATH}', 
        f'--labels={LABEL_PATH}', 
        '--input-blob=images', 
        '--output-blob=output0', 
        '--threshold=0.3'  # 30% 이상 확신할 때만 감지 (너무 낮으면 오작동, 너무 높으면 못 찾음)
    ])

    # 2. 카메라 및 디스플레이 준비
    camera = jetson.utils.videoSource(CAMERA_DEVICE)
    display = jetson.utils.videoOutput("display://0") # HDMI 모니터 출력

    # 3. 한자 폰트 로드 (크기 32px)
    # 이 부분이 없으면 한자가 ㅁㅁㅁ로 깨집니다.
    font = jetson.utils.cudaFont(font=FONT_PATH, size=32)

    print("🎥 영상 감지 시작! (종료하려면 Ctrl+C)")

    while display.IsStreaming():
        # 이미지 캡처
        img = camera.Capture()

        if img is None:
            continue

        # 4. 객체 검출 수행
        # overlay="box" : 박스만 그리고, 기본 글씨는 그리지 않음 (우리가 따로 그릴 거니까)
        detections = net.Detect(img, overlay="box")

        # 5. 감지된 물체마다 한자 라벨 그리기
        for d in detections:
            # ClassID를 이용해 한자 가져오기
            class_name = net.GetClassDesc(d.ClassID)
            
            # 화면에 표시할 텍스트 (예: 丁 95.2%)
            text = f"{class_name} {d.Confidence * 100:.1f}%"
            
            # 박스 왼쪽 상단(d.Left, d.Top)에 글씨 그리기
            # 색상: White(글씨), Gray40(배경)
            font.OverlayText(img, img.width, img.height, 
                             text, int(d.Left), int(d.Top) - 35, 
                             (255, 255, 255, 255), (100, 100, 100, 200))

        # 6. 화면 출력 및 FPS 표시
        display.Render(img)
        display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

if __name__ == '__main__':
    main()