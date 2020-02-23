import torch
from torch.autograd import Variable
from models.utils import strLabelConverter, resizePadding
from PIL import Image
import sys
import models.crnn as crnn
import argparse
from torch.nn.functional import softmax
import numpy as np
import time, os
import cv2
from loader import DatasetLoader
import models.utils as utils
import matplotlib.pyplot as plt
import config

#output_dir = 'outputs/train_' + training_time
#ckpt_prefix=config.ckpt_prefix
img_dir = config.test_dir
pretrained = config.pretrained_test
imgW = config.imgW
imgH = config.imgH
gpu = config.gpu_test
alphabet_name = config.alphabet_name
workers = config.workers_test
batch_size = config.batch_size_test
alphabet = open(alphabet_name).read().rstrip()
nclass = len(alphabet) + 1
nc = 3
debug=True
workers = 4

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def predict(dir, batch_sz, max_iter=1000):
    print('Init CRNN classifier')
    image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
    text = torch.IntTensor(batch_sz * 5)
    length = torch.IntTensor(batch_sz)
    model = crnn.CRNN2(imgH, nc, nclass, 256)
    if gpu != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        model = model.cuda()
        image = image.cuda()
    print('loading pretrained model from %s' % pretrained)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    converter = strLabelConverter(alphabet, ignore_case=False)
    loader = DatasetLoader(dir, imgW, imgH, train_file= 'train', test_file= 'test')
    test_loader = loader.test_loader(batch_sz, num_workers=workers)
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    print('Start predict')
    # for p in crnn.parameters():
    #     p.requires_grad = False
    model.eval()
    val_iter = iter(test_loader)
    max_iter = min(max_iter, len(test_loader))
    print('Number of samples', max_iter)
    begin = time.time()
    with torch.no_grad():
        for i in range(max_iter):
            data = val_iter.next()
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)

            preds = model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            #print(cpu_texts[0])
            print('\n    ',raw_pred, '\n =>', sim_pred,'\ngt:', cpu_texts[0])
            if debug:
                cv_img= cpu_images[0].permute(1, 2, 0).numpy()
                cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                #cv_img_resize=cv2.resize(cv_img_bgr,cv_img_resize,)
                cv2.imshow('result', cv_img_bgr)
                cv2.waitKey(0)
    end = time.time()
    processing_time = end - begin
    print('Processing time:',processing_time)
    print('Speed:', (max_iter*batch_sz) / processing_time, 'fps')


if __name__== "__main__":
    predict(img_dir, batch_size)

