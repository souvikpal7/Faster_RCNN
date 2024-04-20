import torch
import albumentations as A
import cv2
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))
        self.A_transform = A.Compose(
            [
                A.Resize(512, 512, always_apply=True),
                A.HorizontalFlip(p=0.25),
                A.ShiftScaleRotate(p=0.1)
            ],
            bbox_params=A.BboxParams(format='pascal_voc')
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


        # Load annotation from .pkl file
        with open(annotation_path, 'rb') as f:
            annotation = pickle.load(f)

        # Apply transformations
        transformed = self.A_transform(image=img_rgb, bboxes=annotation)
        image = np.transpose(transformed['image'], (2,0,1))
        annotation = np.array(annotation)
        image = torch.from_numpy(image)
        annotation = torch.from_numpy(annotation)

        return image, annotation