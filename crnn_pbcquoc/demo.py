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

img_dir='/data/dataset/ocr_dataset/InkData_line_processed'
img_dir='/data/dataset/cinnamon_data/0825_DataSamples'
img_path=''
alphabet_path='data/char_backup'
#model_path='expr/netCRNN_100.pth'
model_path='outputs/train_2020-02-20_09-03/AICR_pretrained_30.pth'
debug=True
width = 512
height = 32

parser = argparse.ArgumentParser()
parser.add_argument('--img', default=img_path, help='path to img')
parser.add_argument('--alphabet', default=alphabet_path, help='path to vocab')
parser.add_argument('--model', default=model_path, help='path to model')
parser.add_argument('--imgW', type=int, default=width, help='path to model')
parser.add_argument('--imgH', type=int, default=height, help='path to model')

opt = parser.parse_args()


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

alphabet = open(opt.alphabet).read().rstrip()
nclass = len(alphabet) + 1
nc = 3

model = crnn.CRNN(32, nc, nclass, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % opt.model)
model.load_state_dict(torch.load(opt.model, map_location='cpu'))

converter = strLabelConverter(alphabet, ignore_case=False)

begin = time.time()

list_files = get_list_file_in_folder(img_dir)
total_file = len(list_files)
print('Predict:',total_file,'files')

for file in list_files:
    img_path = os.path.join(img_dir, file)
    image = Image.open(img_path).convert('RGB')
    image = resizePadding(image, opt.imgW, opt.imgH)

    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    values, prob = softmax(preds, dim=-1).max(2)
    preds_idx = (prob > 0).nonzero()
    sent_prob = values[preds_idx[:, 0], preds_idx[:, 1]].mean().item()

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print(raw_pred, '=>', sim_pred)
    if debug:
        img = cv2.imread(img_path)
        cv2.imshow('result',img)
        cv2.waitKey(0)


end = time.time()
processing_time = end - begin
print('Processing time:',processing_time)
print('Speed:', total_file/processing_time, 'fps')


