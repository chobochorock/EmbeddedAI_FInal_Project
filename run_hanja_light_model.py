import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys

# ==========================================
# [설정]
ENGINE_PATH = "best_fp16.engine"  # 방금 만든 엔진 파일
LABEL_PATH  = "classes.txt"       # 라벨 파일
INPUT_SIZE  = 640                 # 모델 입력 크기
CONF_THRESH = 0.4                 # 탐지 기준 점수
# ==========================================

# TensorRT 로거 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        # 1. 엔진 파일 로드
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 2. 메모리 할당 (Host & Device)
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Host(CPU) 메모리, Device(GPU) 메모리 할당
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        # 입력 데이터 복사 (CPU -> GPU)
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 추론 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 결과 복사 (GPU -> CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

# GStreamer 파이프라인 (확대/Crop 적용 버전)
def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=0):
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
    # 클래스 로드
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    
    print("🚀 TensorRT 엔진 로딩 중...")
    trt_model = TRTWrapper(ENGINE_PATH)
    print("✅ 로딩 완료!")

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened(): sys.exit("카메라 실패")

    print("실행 시작! (종료: q)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 전처리
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 2. 추론
        output = trt_model.infer(img)[0] # YOLOv5 결과는 보통 1개의 배열로 나옴

        # 3. 후처리 (YOLOv5 Output Parsing)
        # Output shape: (1, 25200, 5+Class) flattened -> reshape 필요
        # 1~10 클래스면 (1, 25200, 15) 형태
        
        output = output.reshape(1, -1, 5 + len(classes))
        output = output[0] # (25200, 15)
        
        boxes, scores, class_ids = [], [], []
        
        # Confidence Threshold 필터링
        # (Numpy 연산으로 속도 최적화)
        conf_mask = output[:, 4] > CONF_THRESH
        detections = output[conf_mask]
        
        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            
            # 최종 스코어 = 객체확률 * 클래스확률
            final_score = confidence * class_score
            
            if final_score > CONF_THRESH:
                x, y, w, h = det[0:4]
                # 좌표 복원 (0~1 -> 0~640)
                left = int((x - 0.5 * w) * INPUT_SIZE)
                top = int((y - 0.5 * h) * INPUT_SIZE)
                width = int(w * INPUT_SIZE)
                height = int(h * INPUT_SIZE)
                
                boxes.append([left, top, width, height])
                scores.append(float(final_score))
                class_ids.append(class_id)

        # 4. NMS 및 그리기
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, 0.45)
        
        if len(indices) > 0:
            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                left, top, w, h = box
                
                label = f"{classes[class_ids[idx]]} {scores[idx]:.2f}"
                cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
                # 한글/한자 출력은 PIL 코드 추가 필요 (일단 기본 확인용)
                cv2.putText(frame, str(class_ids[idx]), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("TensorRT FP16", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()