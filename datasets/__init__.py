from os.path import join
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100
import torchvision.transforms as transforms

DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}


def build_transform(train, img_size, crop, flip):
    transform = []
    transform.append(transforms.Resize((img_size, img_size)))
    transform.append(transforms.Pad(crop)) # why padding?
    if train:
        transform.append(transforms.RandomCrop((img_size, img_size)))
        if flip: transform.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        transform.append(transforms.CenterCrop((img_size, img_size)))
    transform = transforms.Compose(transform)
    return transform


def build_data(data_name, train, trigger, transform):
    # here to enter the directory of datasets
    if data_name=='cifar10':
        data = DATASETS[data_name](root='XXX', train=train, trigger=trigger, transform=transform) 
    elif data_name=='cifar100':
        data = DATASETS[data_name](root='XXX', train=train, trigger=trigger, transform=transform)
    return data
