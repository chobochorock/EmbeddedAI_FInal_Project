import cv2
import numpy as np
import onnxruntime as ort
import sys
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# [설정]
ONNX_MODEL_PATH = "hanja_yolo3/best6.onnx"
LABEL_PATH      = "./hanja_yolo3/classes.txt"
INPUT_SIZE      = 640
CONF_THRESH     = 0.4
FONT_PATH       = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# ==========================================

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=0):
    # 안정적인 기본 파이프라인 사용
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

def main():
    # 1. 클래스 로드
    try:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
    except:
        print("⚠️ classes.txt를 찾을 수 없습니다.")
        classes = []

    # 2. 폰트 로드
    try:
        font = ImageFont.truetype(FONT_PATH, 30)
    except:
        font = ImageFont.load_default()

    # 3. 모델 로드
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
        if not ret:
            print("❌ 카메라 데이터 수신 실패")
            break

        # ------------------------------------------------
        # [전처리]
        # ------------------------------------------------
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(img, axis=0)

        # ------------------------------------------------
        # [추론]
        # ------------------------------------------------
        outputs = session.run([output_name], {input_name: blob})[0]

        # ------------------------------------------------
        # [후처리] 박스 그리기 (주석 해제됨)
        # ------------------------------------------------
        predictions = outputs[0]
        
        boxes = []
        scores = []
        class_ids = []

        x_factor = frame.shape[1] / INPUT_SIZE
        y_factor = frame.shape[0] / INPUT_SIZE

        # 신뢰도 필터링
        conf_mask = predictions[:, 4] > CONF_THRESH
        detections = predictions[conf_mask]

        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            final_score = confidence * class_score
            
            if final_score > CONF_THRESH:
                x, y, w, h = det[0:4]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                scores.append(float(final_score))
                class_ids.append(int(class_id))

        # NMS (겹친 박스 제거)
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, 0.45)

        # 화면에 그리기
        if len(indices) > 0:
            # 한자 출력을 위해 PIL로 변환
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                left, top, w, h = box[0], box[1], box[2], box[3]
                
                # 라벨 텍스트
                if class_ids[idx] < len(classes):
                    label = f"{classes[class_ids[idx]]} {scores[idx]:.2f}"
                else:
                    label = f"ID:{class_ids[idx]} {scores[idx]:.2f}"

                # 박스와 글씨 그리기
                draw.rectangle([left, top, left+w, top+h], outline=(0, 255, 0), width=3)
                draw.text((left, top - 30), label, font=font, fill=(0, 255, 0))
            
            # 다시 OpenCV 포맷으로 변환
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # ------------------------------------------------
        
        cv2.imshow("Hanja Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()