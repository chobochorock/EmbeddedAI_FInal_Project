import cv2
import numpy as np
import onnxruntime as ort
import sys
from PIL import Image, ImageDraw, ImageFont

# 설정
ONNX_MODEL_PATH = "best.onnx"
LABEL_PATH = "labels.txt"
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
CONFIDENCE_THRESHOLD = 0.4
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def put_text_hanja(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 폰트 파일 없으면 기본 폰트(한자 안나옴) 사용
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    classes = load_classes(LABEL_PATH)
    print(f"총 {len(classes)}개의 클래스 로드")

    # [핵심 변경] OpenCV DNN 대신 ONNX Runtime 사용
    print("ONNX Runtime으로 모델 로딩 중...")
    
    # GPU(CUDA) 사용 설정
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        print("ONNX Runtime 로드 성공! (GPU 가속 시도)")
    except Exception as e:
        print(f"로딩 실패: {e}")
        return

    # 입력/출력 이름 알아내기
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # cap = cv2.VideoCapture(0) # 혹은 GStreamer 문자열
    # 1. GStreamer 파이프라인 문자열 생성 함수
    def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=128,
        display_height=128,
        framerate=30,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    # 2. 카메라 열기 (GStreamer 모드 사용)
    print("📸 CSI 카메라를 GStreamer로 여는 중...")
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        print("✅ 카메라 연결 성공!")
    else:
        print("❌ 카메라 연결 실패: 데몬을 재시작했는지 확인해주세요.")
        sys.exit()
    
    # if not cap.isOpened():
    #     print("카메라 열기 실패")
    #     sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 전처리 (OpenCV DNN과 약간 다름)
        # BGR -> RGB 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        # 0~255 -> 0.0~1.0
        img = img.astype(np.float32) / 255.0 # float32
        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        img = img.transpose(2, 0, 1)
        blob = np.expand_dims(img, axis=0)

        # 추론 실행 (ONNX Runtime)
        outputs = session.run([output_name], {input_name: blob})[0]

        # 후처리 (이후 로직은 기존과 유사하나 데이터 형태에 따라 조정 필요)
        # YOLOv8 Output: (1, 4+cls, 8400)
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        x_factor = frame.shape[1] / INPUT_WIDTH
        y_factor = frame.shape[0] / INPUT_HEIGHT

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)
            
            if max_score >= CONFIDENCE_THRESHOLD:
                # 좌표 계산 등 기존과 동일...
                class_id = max_class_loc[1]
                box = outputs[0][i][:4]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        # NMS 및 그리기 (기존 코드와 동일)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
        
        if len(indices) > 0:
            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                left, top, width, height = box[0], box[1], box[2], box[3]
                
                # 라벨 텍스트 생성
                if class_ids[idx] < len(classes):
                    label_text = f"{classes[class_ids[idx]]}" # 점수 빼고 글자만 크게
                else:
                    label_text = "Unknown"

                # 1. 박스 그리기 (OpenCV 사용)
                cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
                
                # 2. [핵심 변경] 한자 그리기 (PIL 사용)
                # 기존 cv2.putText(...) 줄을 지우고 아래 줄로 교체하세요.
                frame = put_text_hanja(frame, label_text, (left, top + 30), FONT_PATH, 30, (0, 255, 0))

        cv2.imshow("Hanja Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()