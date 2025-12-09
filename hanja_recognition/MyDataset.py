import os
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms.v2 import GaussianNoise


class MyDataset(Dataset):
    def __init__(self, path: str='data', split: str='train', transform_image=None):
        self.sigma_noise = 1.0
        if split.lower() == 'mytrain':
            split = 'train'

            self.transform_image = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                # transforms.RandomRotation(degrees=90), 
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),              # data range = [0.0, 1.0]
                transforms.Normalize([0.5], [0.5]), # data range = [-1.0, 1.0]
                # GaussianNoise(mean=0.0, sigma=self.sigma_noise, clip=False),
            ]) if transform_image is None else transform_image

        self.dir_data   = os.path.join(path)
        self.file_image = glob.glob(self.dir_data + '/*.png')

        self.raw_label = [os.path.basename(path).split('_')[0] for path in self.file_image]
        self.classes = sorted(list(set(self.raw_label)))
        
        # 3. [핵심] 문자를 숫자로(char -> idx), 숫자를 문자로(idx -> char) 바꾸는 사전 생성
        self.char2idx = {char: i for i, char in enumerate(self.classes)}
        self.idx2char = {i: char for i, char in enumerate(self.classes)}

        self.data = []
        self.labels = []
        for i, file_path in enumerate(tqdm(self.file_image)):
            # 이미지를 열고, 메모리 절약을 위해 미리 'L'(흑백) 모드로 변환
            img = Image.open(file_path).convert('L') 
            
            # transform이 있다면 여기서 미리 적용할 수도 있지만,
            # Data Augmentation(회전, 노이즈 등)이 매번 달라져야 한다면
            # 여기서는 'ToTensor' 전까지만 하거나, 원본 PIL 이미지를 저장하는 게 좋습니다.
            # (여기서는 원본 PIL 객체 자체를 저장하거나, numpy로 변환해 저장 추천)
            self.data.append(img)
            
            # 라벨도 미리 숫자로 바꿔서 저장
            label_char = self.raw_label[i]
            self.labels.append(self.char2idx[label_char])

    def __len__(self):
        return len(self.file_image)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_image      = self.file_image[idx]
        label_idx       = self.labels[idx]
        image           = Image.open(file_image)

        if self.transform_image is not None:
            image = self.transform_image(image)

        return (image, label_idx)