import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
from PIL import Image, ImageDraw, ImageFont # [추가] 한자 출력을 위해 필수!

# ==========================================
# [설정] 파일 경로 확인!
ENGINE_PATH = "best_fp16.engine"  # TensorRT 엔진 파일
LABEL_PATH  = "classes.txt"       # 라벨 파일
FONT_PATH   = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc" # 한자 폰트 (없으면 기본폰트 사용)
INPUT_SIZE  = 640                 # 학습할 때 이미지 크기
CONF_THRESH = 0.4                 # 탐지 기준 점수
# ==========================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
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
    # 2배 줌인 (Crop) 설정
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
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 2. 폰트 로드 [추가된 부분]
    try:
        font = ImageFont.truetype(FONT_PATH, 30)
    except:
        print("⚠️ 폰트 파일을 찾을 수 없어 기본 폰트를 사용합니다. (한자가 안 보일 수 있음)")
        font = ImageFont.load_default()

    print("🚀 TensorRT 엔진 로딩 중...")
    trt_model = TRTWrapper(ENGINE_PATH)
    print("✅ 로딩 완료! 실행합니다.")

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened(): sys.exit("❌ 카메라 실패")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 전처리
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 추론 (TensorRT)
        output = trt_model.infer(img)[0]
        
        # 후처리
        output = output.reshape(1, -1, 5 + len(classes))
        output = output[0]
        
        boxes, scores, class_ids = [], [], []
        
        # Numpy 가속 필터링
        conf_mask = output[:, 4] > CONF_THRESH
        detections = output[conf_mask]
        
        for det in detections:
            confidence = det[4]
            class_probs = det[5:]
            class_id = np.argmax(class_probs)
            final_score = confidence * class_probs[class_id]
            
            if final_score > CONF_THRESH:
                x, y, w, h = det[0:4]
                left = int((x - 0.5 * w) * INPUT_SIZE)
                top = int((y - 0.5 * h) * INPUT_SIZE)
                width = int(w * INPUT_SIZE)
                height = int(h * INPUT_SIZE)
                
                boxes.append([left, top, width, height])
                scores.append(float(final_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, 0.45)
        
        # [핵심 변경] 그리기 단계 (PIL 사용)
        if len(indices) > 0:
            # OpenCV 이미지를 PIL로 변환 (한자를 그리기 위해)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            for i in indices:
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                left, top, w, h = box
                
                # 라벨 텍스트 (한자 포함)
                label = f"{classes[class_ids[idx]]} {scores[idx]:.0%}"
                
                # 박스 그리기
                draw.rectangle([left, top, left+w, top+h], outline=(0, 255, 0), width=3)
                # 글씨 그리기 (배경 박스 + 글자)
                draw.rectangle([left, top-35, left+100, top], fill=(0, 255, 0)) # 글자 배경
                draw.text((left+5, top-35), label, font=font, fill=(255, 255, 255)) # 흰색 글씨

            # 다시 OpenCV 포맷으로 변환 (화면 출력을 위해)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Hanja TensorRT", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()