import cv2
import numpy as np
import sys

# ==========================================
# 설정 (여기를 수정하세요)
# ==========================================
ONNX_MODEL_PATH = "best.onnx"   # PC에서 가져온 ONNX 파일
LABEL_PATH      = "labels.txt"  # 한자 리스트 파일
INPUT_WIDTH     = 640           # 학습시킬 때 이미지 크기
INPUT_HEIGHT    = 640
SCORE_THRESHOLD = 0.4           # 이 점수보다 낮으면 무시
NMS_THRESHOLD   = 0.45          # 박스 겹침 제거 기준
CONFIDENCE_THRESHOLD = 0.4      # 객체일 확률 기준

# 카메라 설정 (CSI: 0, USB: 1 등, 안되면 문자열 "/dev/video0" 시도)
CAMERA_ID = 0 
# ==========================================

def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def main():
    # 1. 클래스 로드
    classes = load_classes(LABEL_PATH)
    print(f"총 {len(classes)}개의 한자 클래스 로드 완료")

    # 2. OpenCV DNN으로 ONNX 모델 로드
    print("모델을 불러오는 중...")
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)

    # [핵심] Jetson Nano의 GPU(CUDA)를 사용하도록 강제 설정
    # 이 설정이 없으면 CPU로 돌아가서 엄청 느립니다.
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("CUDA(GPU) 가속 활성화 완료!")
    except:
        print("경고: CUDA를 사용할 수 없습니다. CPU로 실행합니다. (느릴 수 있음)")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 3. 카메라 열기
    # Jetson CSI 카메라는 GStreamer 파이프라인 문자열이 필요할 수 있음.
    # 일단 간단하게 0번으로 시도해보고 안되면 GStreamer 문자열 사용 권장.
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # CSI 카메라용 GStreamer 문자열 (위 코드로 안 될 경우 주석 해제 후 사용)
    # gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    # cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        sys.exit()

    print("실행 시작! (종료: q)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 이미지 전처리 (Blob 변환)
        # YOLO는 0~255 픽셀값을 0~1로 정규화(1/255.0)해서 넣어야 함
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)

        # 4. 추론 실행
        outputs = net.forward()

        # YOLOv8 출력 처리
        # 출력 형태는 보통 (1, 4+ClassNum, 8400) 형태임 -> Transpose 필요
        # YOLOv5의 경우 (1, 25200, 5+ClassNum) 형태일 수 있음. 
        # 아래 코드는 YOLOv8/v5 일반적인 출력을 파싱합니다.
        
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # 원본 이미지 비율 계산
        x_factor = frame.shape[1] / INPUT_WIDTH
        y_factor = frame.shape[0] / INPUT_HEIGHT

        for i in range(rows):
            # YOLOv8 기준: [x, y, w, h, class1_score, class2_score, ...]
            classes_scores = outputs[0][i][4:]
            _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)
            class_id = max_class_loc[1]
            
            if max_score >= CONFIDENCE_THRESHOLD:
                box = outputs[0][i][:4]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        # 5. NMS (겹친 박스 제거)
        indices = cv2.dnn.NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD)

        # 6. 결과 그리기
        for i in indices:
            # cv2.dnn.NMSBoxes 결과가 버전에 따라 리스트일수도, 튜플일수도 있음
            idx = i if isinstance(i, int) else i[0]
            
            box = boxes[idx]
            left, top, width, height = box[0], box[1], box[2], box[3]
            
            label = f"{classes[class_ids[idx]]}: {scores[idx]:.2f}"
            
            # 박스
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            # 글씨 (OpenCV 기본 폰트는 한자 출력이 안 됩니다! 네모로 나옴)
            # 한자 출력을 원하면 여기에 PIL 코드를 섞어야 합니다.
            # 일단은 영어 클래스 번호나 이름이 나오는지 확인하세요.
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Pure OpenCV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()