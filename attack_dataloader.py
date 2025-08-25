import csv
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)


def get_transform(opt, train=True, c=0, k=0):
    if opt.dataset == "imagenet200":
        transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4) if train else transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_list)
    elif opt.dataset == "tinyimagenet200":
        transform_list = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.448, 0.397), (0.276, 0.268, 0.281))
        ]
        return transforms.Compose(transform_list)
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
        if opt.dataset != "mnist":
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
        if opt.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = os.path.join(self.data_folder, format(c, "05d"))
            gt_path = os.path.join(prefix, "GT-" + format(c, "05d") + ".csv")
            with open(gt_path) as gtFile:
                gtReader = csv.reader(gtFile, delimiter=";")
                next(gtReader)
                for row in gtReader:
                    images.append(os.path.join(prefix, row[0]))
                    labels.append(int(row[7]))
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        gt_path = os.path.join(self.data_folder, "GT-final_test.csv")
        with open(gt_path) as gtFile:
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(os.path.join(self.data_folder, row[0]))
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class ImageNet(data.Dataset):
    def __init__(self, opt, train=True, transform=None):
        super(ImageNet, self).__init__()
        dataset_dir = os.path.join(opt.data_root, opt.dataset)
        if train:
            self.data_folder = os.path.join(dataset_dir, 'train')
        else:
            self.data_folder = os.path.join(dataset_dir, 'test')
        self.data = datasets.ImageFolder(self.data_folder, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label


def get_dataloader(opt, train=True, c=0, k=0):
    transform = get_transform(opt, train, c=c, k=k)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "imagenet200":
        dataset = ImageNet(opt, train, transform)
    elif opt.dataset == "tinyimagenet200":
        dataset = ImageNet(opt, train, transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader
