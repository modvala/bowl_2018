import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import prepare_data

data_path = Path('input')
binary_factor = 256


class bowl_18_dataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        img = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return to_float_tensor(img), torch.from_numpy(mask).long()
        else:
            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    img = cv2.imread(str('input/croped_train/')+str(path)+str('/img.png'))
    #print(img.shape)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def load_mask(path, problem_type):
    if problem_type == 'binary':
        factor = binary_factor

    mask = cv2.imread(str('input/croped_train/')+str(path)+str('/mask.png'), 0)
    #print(mask.shape)
    return (mask / factor).astype(np.uint8)
