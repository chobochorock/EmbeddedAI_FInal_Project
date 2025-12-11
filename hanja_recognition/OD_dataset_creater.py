import os
import random
import glob
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from tqdm import tqdm
import shutil

# ==========================================
# [설정]
SOURCE_DATA_DIR = "../Traditional-Chinese-Handwriting-Dataset/data/cleaned_data(50_50)"
OUTPUT_DIR = "./yolo_dataset_natural"
CANVAS_SIZE = 640
TOTAL_IMAGES = 2000
MIN_OBJS = 3
MAX_OBJS = 8
TARGET_CHARS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
# ==========================================

def get_natural_paper_bg(size):
    """
    작은 노이즈를 확대하여 부드러운 음영(구름 효과)을 만들고,
    종이 색감(미색)을 입힙니다.
    """
    # 1. 아주 작은 노이즈 생성 (예: 16x16)
    small_size = 32
    noise_array = np.random.randint(200, 255, (small_size, small_size), dtype=np.uint8)
    noise_img = Image.fromarray(noise_array, mode='L')
    
    # 2. 크게 확대 (BICUBIC) -> 픽셀이 뭉개지면서 부드러운 그림자가 됨
    texture = noise_img.resize((size, size), resample=Image.BICUBIC)
    
    # 3. 그레이스케일을 RGB로 변환 (종이 색감을 위해)
    texture = texture.convert("RGB")
    
    # 4. 약간의 누런 종이 색감(미색) 추가
    # 노란색 레이어를 만들어서 섞어줌
    paper_color = Image.new("RGB", (size, size), (255, 250, 240)) # 상아색
    final_bg = ImageChops.multiply(texture, paper_color)
    
    return final_bg

def is_overlap(new_box, existing_boxes):
    nx, ny, nw, nh = new_box
    for (ex, ey, ew, eh) in existing_boxes:
        if not (nx + nw < ex or nx > ex + ew or ny + nh < ey or ny > ey + eh):
            return True
    return False

def convert_to_yolo_bbox(canvas_size, x, y, w, h):
    dw = 1. / canvas_size
    dh = 1. / canvas_size
    x_center = (x + w / 2.0) * dw
    y_center = (y + h / 2.0) * dh
    width = w * dw
    height = h * dh
    return x_center, y_center, width, height

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(f"{OUTPUT_DIR}/train/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/train/labels", exist_ok=True)

    # 이미지 분류
    char_images = {char: [] for char in TARGET_CHARS}
    all_files = glob.glob(f"{SOURCE_DATA_DIR}/*.png")
    for p in all_files:
        filename = os.path.basename(p)
        char = filename.split('_')[0]
        if char in char_images:
            char_images[char].append(p)

    class_to_id = {char: i for i, char in enumerate(TARGET_CHARS)}
    
    with open(f"{OUTPUT_DIR}/classes.txt", "w", encoding="utf-8") as f:
        for char in TARGET_CHARS:
            f.write(char + "\n")

    print(f"🚀 자연스러운 종이 질감 데이터셋 생성 시작 ({TOTAL_IMAGES}장)...")

    for i in tqdm(range(TOTAL_IMAGES)):
        # (1) 자연스러운 종이 배경 생성
        canvas = get_natural_paper_bg(CANVAS_SIZE)
        
        num_objs = random.randint(MIN_OBJS, MAX_OBJS)
        existing_boxes = []
        label_lines = []

        for _ in range(num_objs):
            char_choice = random.choice(TARGET_CHARS)
            if not char_images[char_choice]: continue
            
            img_path = random.choice(char_images[char_choice])
            try:
                # 흑백으로 염
                src_img = Image.open(img_path).convert("L")
                
                # [핵심] 배경 흰색 날리기 (Threshold) & 투명화 준비
                # 글자 부분은 검정(0), 배경은 흰색(255)이라고 가정
                # 색상 반전 -> 글자가 흰색(255), 배경이 검정(0)이 됨
                # 이걸 마스크로 써서 합성하거나, Multiply 모드 사용
                
                # 여기서는 가장 자연스러운 'Multiply(곱하기)' 방식 사용을 위해 RGB 변환
                src_img_rgb = Image.new("RGB", src_img.size, (255, 255, 255))
                src_img_rgb.paste(src_img, (0,0)) # 흑백 이미지를 RGB로
                
            except: continue
            
            w, h = src_img.size
            placed = False
            
            for _ in range(50):
                x = random.randint(0, CANVAS_SIZE - w)
                y = random.randint(0, CANVAS_SIZE - h)
                
                if not is_overlap((x, y, w, h), existing_boxes):
                    # [핵심] 자연스러운 합성 (Multiply)
                    # 1. 캔버스에서 해당 위치 부분만 잘라냄
                    crop = canvas.crop((x, y, x+w, y+h))
                    # 2. 잘라낸 배경과 글자 이미지를 '곱하기' 모드로 합성
                    # (흰색은 투명해지고 검은 글씨만 배경에 묻어남)
                    blended = ImageChops.multiply(crop, src_img.convert('RGB'))
                    # 3. 합성된 조각을 다시 캔버스에 붙임
                    canvas.paste(blended, (x, y))
                    
                    existing_boxes.append((x, y, w, h))
                    
                    cid = class_to_id[char_choice]
                    cx, cy, bw, bh = convert_to_yolo_bbox(CANVAS_SIZE, x, y, w, h)
                    label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    placed = True
                    break
        
        file_stem = f"{i:06d}"
        canvas.save(f"{OUTPUT_DIR}/train/images/{file_stem}.jpg")
        with open(f"{OUTPUT_DIR}/train/labels/{file_stem}.txt", "w") as f:
            f.write("\n".join(label_lines))

    print(f"✅ 완료! 확인해보세요: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()