import cv2
import numpy as np
import onnxruntime as ort
import sys
import time

# ==========================================
ONNX_MODEL_PATH = "hanja_yolo3/best.onnx"
INPUT_SIZE      = 640
# ==========================================

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=640, framerate=30, flip_method=0):
    # [진단] 복잡한 Crop 기능을 끄고 가장 단순한 파이프라인으로 테스트
    # 만약 이게 되면 아까 그 Crop 설정이 문제였던 것입니다.
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
    print("STEP 1: ONNX 세션 초기화 중...")
    try:
        # CPU로만 먼저 테스트 (CUDA 문제인지 확인)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print("✅ ONNX 모델 로드 성공")
    except Exception as e:
        print(f"❌ ONNX 모델 로드 실패: {e}")
        return

    print("STEP 2: 카메라 파이프라인 생성 중...")
    pipeline = gstreamer_pipeline()
    print(f"Pipeline: {pipeline}")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("✅ 카메라 장치 열기 성공 (isOpened=True)")
    else:
        print("❌ 카메라 장치를 열 수 없습니다. 데몬을 재시작해보세요.")
        return

    print("STEP 3: 프레임 읽기 루프 진입")
    try:
        while True:
            # 시간 측정
            start = time.time()
            
            print("   -> 프레임 읽기 시도...", end="")
            ret, frame = cap.read()
            
            if not ret:
                print("\n⚠️ [치명적 문제] ret=False가 반환되었습니다.")
                print("   카메라가 열리긴 했으나 데이터를 보내지 않습니다.")
                print("   원인 1: GStreamer 파이프라인 설정 오류")
                print("   원인 2: 카메라 하드웨어 연결 불량")
                break
            print("성공! ", end="")

            # 전처리
            print("전처리... ", end="")
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
            blob = np.expand_dims(img, axis=0)

            # 추론
            print("추론(Inference)... ", end="")
            _ = session.run([output_name], {input_name: blob})[0]
            print(f"완료! ({time.time()-start:.3f}초)")

            cv2.imshow("Test", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"\n❌ 실행 중 에러 발생: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()