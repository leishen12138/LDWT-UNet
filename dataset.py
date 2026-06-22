import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


CLASSES = ['fish', 'sea turtle', 'sea horse', 'sea snake', 'shark', 'whale', 'dolphin']
class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}


class ToTensor(object):
    def __call__(self, data):
        data['image'] = F.to_tensor(data['image'])
        data['label'] = F.to_tensor(data['label'])
        return data   # 保留所有键

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        data['image'] = F.resize(data['image'], self.size)
        data['label'] = F.resize(data['label'], self.size, interpolation=InterpolationMode.BICUBIC)
        return data

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.hflip(data['image'])
            data['label'] = F.hflip(data['label'])
        return data

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.vflip(data['image'])
            data['label'] = F.vflip(data['label'])
        return data

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        data['image'] = F.normalize(data['image'], self.mean, self.std)
        return data
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, txt_root=None):
        self.txt_root = txt_root if txt_root is not None else image_root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        filename = os.path.basename(self.images[idx])
        txt_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(self.txt_root, txt_filename)
        with open(txt_path, 'r') as f:
            label_str = f.read().strip().lower()
        # 类别索引
        label_idx = class_to_idx.get(label_str, -1)
        if label_idx == -1:
            raise ValueError(f"未知类别 '{label_str}' 在文件 {txt_path}")
        data = {'image': image, 'label': label, 'label_idx': label_idx}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size, txt_root=None):
        self.txt_root = txt_root if txt_root is not None else image_root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)
        filename = os.path.basename(self.images[self.index])
        txt_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(self.txt_root, txt_filename)
        with open(txt_path, 'r') as f:
            label_str = f.read().strip().lower()
        label_idx = class_to_idx.get(label_str, -1)
        if label_idx == -1:
            label_idx = 0  # 默认 fish

        name = self.images[self.index]
        self.index += 1
        return image, gt, label_idx, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')