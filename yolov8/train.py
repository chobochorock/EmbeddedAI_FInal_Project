from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    print("학습 시작")
    model.train(
        data='data.yaml', 
        epochs=30, 
        imgsz=128,     
        batch=64,      
        device=0,      
        pretrained=True 
    )

    print("변환 중")
    path = model.export(format='onnx', dynamic=False, imgsz=128) 

    print(f"변환 완료 : {path}")

if __name__ == '__main__':
    main()