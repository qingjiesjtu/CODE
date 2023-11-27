import os
import numpy as np
import pickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class CIFAR100(data.Dataset):
    base_folder = 'cifar-100-python'
    train_list = [
        ['train', 'c99cafc152244af753f735de768cd75f'],
    ]
    test_list = [
        ['test', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, trigger=None, transform=None):
        super(CIFAR100, self).__init__()
        self.root = root
        self.trigger = trigger
        self.transform = transform
        file_list = self.train_list if train else self.test_list
        self.data, self.targets = [], []
        for file_name, checksum in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        backdoor, source = 0, target # backdoor表示样本是否投毒
        img = Image.fromarray(img)
        if self.trigger is not None: img, target, backdoor = self.trigger(img, target, backdoor, idx)
        if self.transform is not None: img = self.transform(img)
        img = self.toTensor(img)
        return img, target, backdoor, source, idx

    def __len__(self):
        return self.data.shape[0]
