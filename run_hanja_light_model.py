import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# [설정]
ENGINE_PATH = "hanja_yolo3/best6_light.engine"
LABEL_PATH  = "hanja_yolo3/classes.txt"
# 우분투 기본 한글/한자 폰트 경로 예시 (나눔고딕 등 추천)
# 만약 파일이 없다면 'NanumGothic.ttf' 등을 프로젝트 폴더에 넣고 경로 수정하세요.
FONT_PATH   = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc" 
INPUT_SIZE  = 640
CONF_THRESH = 0.4
IOU_THRESH  = 0.45
# ==========================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
        except FileNotFoundError:
            sys.exit(f"❌ 엔진 파일을 찾을 수 없습니다: {engine_path}")

        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=0):
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
    except FileNotFoundError:
        sys.exit(f"❌ 라벨 파일을 찾을 수 없습니다: {LABEL_PATH}")

    # 2. 폰트 로드
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        print(f"⚠️ 경고: {FONT_PATH}를 찾을 수 없습니다. 기본 폰트를 사용합니다 (한자 깨짐 가능).")
        font = ImageFont.load_default()

    print("🚀 TensorRT 엔진 로딩 중...")
    trt_model = TRTWrapper(ENGINE_PATH)
    print("✅ 로딩 완료! 카메라를 시작합니다.")

    # GStreamer 파이프라인 (디스플레이 크기와 입력 크기가 다를 수 있음을 대비)
    cap = cv2.VideoCapture(gstreamer_pipeline(display_width=INPUT_SIZE, display_height=INPUT_SIZE), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        sys.exit("❌ 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")

    # FPS 계산용 변수
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        
        # [전처리] 비율 유지를 위한 스케일링 팩터 계산 (현재는 1:1이지만 확장성 고려)
        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE

        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # [추론]
        output = trt_model.infer(img)[0]
        
        # [후처리] YOLO 출력 형상에 맞게 reshape (모델마다 다를 수 있음, 확인 필요)
        # 보통 YOLOv5/v8 export 시: [Batch, Anchors, 5+Classes] or [Batch, 5+Classes, Anchors]
        # 여기서는 작성자분의 코드(flat -> reshape)를 따릅니다.
        output = output.reshape(1, -1, 5 + len(classes))
        output = output[0]

        boxes, scores, class_ids = [], [], []
        
        # 신뢰도 필터링
        conf_mask = output[:, 4] > CONF_THRESH
        detections = output[conf_mask]

        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            final_score = confidence * class_probs[class_id]

            if final_score > CONF_THRESH:
                # 좌표 복원 (0~1 정규화된 값이 아니라 픽셀 값으로 나온다고 가정 - YOLO 버전에 따라 다름)
                # 만약 박스가 이상하게 크다면 아래 로직 확인 필요
                cx, cy, bw, bh = det[0:4]
                
                # 원본 이미지 크기에 맞춰 스케일링
                left = int((cx - 0.5 * bw) * scale_x)
                top = int((cy - 0.5 * bh) * scale_y)
                width = int(bw * scale_x)
                height = int(bh * scale_y)

                boxes.append([left, top, width, height])
                scores.append(float(final_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)

        # [그리기 단계] 박스가 있을 때만 PIL 변환 수행 (속도 최적화)
        if len(indices) > 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                left, top, width, height = box
                
                # 좌표가 이미지 밖으로 나가지 않게 클램핑
                left = max(0, left)
                top = max(0, top)

                # 라벨 생성
                label_text = f"{classes[class_ids[idx]]} {scores[idx]:.0%}"
                
                # 텍스트 사이즈 계산 (배경 박스 크기 자동 조절)
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]

                # 박스 및 텍스트 그리기
                draw.rectangle([left, top, left + width, top + height], outline=(0, 255, 0), width=3)
                draw.rectangle([left, top - text_h - 10, left + text_w + 10, top], fill=(0, 255, 0))
                draw.text((left + 5, top - text_h - 5), label_text, font=font, fill=(255, 255, 255))

            # 다시 OpenCV 포맷으로
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # FPS 출력
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hanja TensorRT", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()