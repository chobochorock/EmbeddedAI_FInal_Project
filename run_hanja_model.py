import cv2
import numpy as np
import onnxruntime as ort
import sys
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# [설정] 본인 환경에 맞게 수정하세요
# ==========================================
ONNX_MODEL_PATH = "hanja_yolo3/best_fp16.onnx"   # PC에서 가져온 ONNX 파일 (opset 10 권장)
LABEL_PATH      = "./hanja_yolo3/classes.txt" # 클래스 이름이 적힌 파일
INPUT_SIZE      = 640           # 학습할 때 사용한 이미지 크기 (640 or 1280 등)
CONF_THRESH     = 0.4           # 탐지 신뢰도 기준
FONT_PATH       = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc" # 한자 폰트 경로
# ==========================================

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=0):
    """
    한자 인식을 위해 중앙 부분을 확대(Crop & Zoom)해서 가져오는 GStreamer 파이프라인
    """
    crop_left, crop_right = 320, 1280-320
    crop_top, crop_bottom = 40, 720-40
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv left=%d right=%d top=%d bottom=%d flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, crop_left, crop_right, crop_top, crop_bottom, flip_method, display_width, display_height)
    )

def main():
    # 1. 클래스 로드
    try:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
    except:
        print("⚠️ classes.txt를 찾을 수 없습니다. 라벨 표시가 안 될 수 있습니다.")
        classes = []

    # 2. 폰트 로드
    try:
        font = ImageFont.truetype(FONT_PATH, 30)
    except:
        font = ImageFont.load_default()

    # 3. ONNX 모델 로드 (GPU 가속 활성화)
    print(f"🚀 ONNX 모델 로딩 중: {ONNX_MODEL_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ 모델 로드 완료!")

    # 4. 카메라 연결
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("❌ 카메라를 열 수 없습니다.")

    print("🎥 실행 시작! (종료: q)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ------------------------------------------------
        # [전처리] YOLOv5 입력 형식에 맞추기
        # ------------------------------------------------
        # 1. Resize
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        # 2. BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. Normalize (0~1) & Transpose (HWC -> CHW)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        # 4. Batch 차원 추가 (1, 3, 640, 640)
        blob = np.expand_dims(img, axis=0)

        # ------------------------------------------------
        # [추론] ONNX Runtime 실행
        # ------------------------------------------------
        outputs = session.run([output_name], {input_name: blob})[0]

        # ------------------------------------------------
        # [후처리] 결과 파싱 (YOLOv5 Output)
        # ------------------------------------------------
        # Output shape: (1, 25200, 5+Class) -> (1, N, 85 등)
        predictions = outputs[0] 

        boxes = []
        scores = []
        class_ids = []

        # 원본 해상도 복원을 위한 비율
        x_factor = frame.shape[1] / INPUT_SIZE
        y_factor = frame.shape[0] / INPUT_SIZE

        # 신뢰도 필터링 (for문 대신 Numpy 연산으로 속도 최적화)
        # confidence(obj_conf) * class_score 가 기준 이상인 것만 필터링
        
        # 4번 인덱스(Objectness)가 임계값보다 큰 것만 1차 필터링
        conf_mask = predictions[:, 4] > CONF_THRESH
        detections = predictions[conf_mask]

        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            
            # 최종 점수
            final_score = confidence * class_score
            
            if final_score > CONF_THRESH:
                x, y, w, h = det[0:4]
                
                # 좌표 복원 (Center_XYWH -> TopLeft_XYWH)
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                scores.append(float(final_score))
                class_ids.append(int(class_id))

        # NMS (겹친 박스 제거)
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, 0.45)

        # ----------------