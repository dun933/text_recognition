import torchvision
import torch
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import argparse
import time
from multiprocessing import cpu_count
import uuid
from models.utils import resizePadding
from torchvision import transforms




def img_loader(path):
    img = Image.open(path).convert('RGB')
    return img


def train_transform(path):
    pass


def test_transform(path):
    pass


def target_loader(path):
    label = open(path.replace('/images/', '/annos/')).read().rstrip('\n')
    return label


def default_flist_reader(flist, label=True):
    imlist = []
    img_exts = ('jpg', 'png', 'JPG', 'PNG')
    with open(flist) as rf:
        for line in rf.readlines():
            impath = line.strip()
            if impath.endswith(img_exts):
                imlabel = os.path.splitext(impath)[0] + '.txt'
                imlist.append((impath, imlabel))

    return imlist


class ImageFileList(data.Dataset):
    def __init__(self, root, flist, transform, label_transform, flist_reader = default_flist_reader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        impath, targetpath = self.imlist[index]
        imgpath = os.path.join(self.root, impath)
        targetpath = os.path.join(self.root, targetpath)

        img = self.transform(imgpath)
        target = self.label_transform(targetpath)

        return img, target

    def __len__(self):
        return len(self.imlist)


class alignCollate(object):
    def __init__(self, imgW, imgH):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images = batch
        images = [resizePadding(image, self.imgW, self.imgH) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images


class TestLoader(object):
    def __init__(self, root, imgW, imgH, img_list): #img_list is list of PIL images
        self.root = root
        self.imgW = imgW
        self.imgH = imgH
        self.test_dataset = ImageFileList(root, img_list, transform=img_loader, label_transform=target_loader)

    def test_loader(self, batch_size, num_workers=4, shuffle=False):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=alignCollate(self.imgW, self.imgH)
        )
        return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='path to root folder')
    parser.add_argument('--train', required=True, help='path to train list')
    parser.add_argument('--val', required=True, help='path to test list')
    opt = parser.parse_args()

    loader = TestLoader(opt.root, opt.train, opt.val, 512, 32)
    for _ in range(100):
        train_loader = iter(loader.train_loader(64, num_workers=cpu_count()))
        i = 0
        while i < len(train_loader):
            start_time = time.time()
            X_train, y_train = next(train_loader)
            elapsed_time = time.time() - start_time

            i += 1
            print(i, elapsed_time, X_train.size())


if __name__ == '__main__':
    main()
