import cv2
import numpy as np
import onnxruntime as ort  # 변경된 부분
import sys

# 설정
ONNX_MODEL_PATH = "best.onnx"
LABEL_PATH = "labels.txt"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4

def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

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

    cap = cv2.VideoCapture(0) # 혹은 GStreamer 문자열
    if not cap.isOpened():
        print("카메라 열기 실패")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 전처리 (OpenCV DNN과 약간 다름)
        # BGR -> RGB 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        # 0~255 -> 0.0~1.0
        img = img.astype(np.float32) / 255.0
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
                
                # 라벨 표시 (영어)
                label = f"Class {class_ids[idx]}: {scores[idx]:.2f}"
                try:
                     # 한자 리스트가 있다면
                     label = f"{classes[class_ids[idx]]} {scores[idx]:.2f}"
                except: pass

                cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("ONNX Runtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()