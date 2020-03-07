import torch
from torch.autograd import Variable
from models.utils import strLabelConverter
import models.crnn as crnn
import models.crnn128 as crnn128
import time, os

import cv2
from loader import ImageFileLoader, alignCollate, NumpyListLoader
import models.utils as utils
import config
from torchvision import transforms

# output_dir = 'outputs/train_' + training_time
# ckpt_prefix=config.ckpt_prefix
img_dir = config.test_dir
test_list = config.test_list
pretrained = config.pretrained_test
imgW = config.imgW
imgH = config.imgH
gpu = config.gpu_test
alphabet_path = config.alphabet_path
workers = config.workers_test
batch_size = config.batch_size_test
label = config.label
debug = config.debug
alphabet = open(alphabet_path).read().rstrip()
nclass = len(alphabet) + 1
nc = 3
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def predict(dir, batch_sz, max_iter=10000):
    print('Init CRNN classifier')
    image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
    model = crnn.CRNN2(imgH, nc, nclass, 256)
    #model = crnn128.CRNN128(imgH, nc, nclass, 256)
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        model = model.cuda()
        image = image.cuda()
    print('loading pretrained model from %s' % pretrained)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    converter = strLabelConverter(alphabet, ignore_case=False)
    val_dataset = ImageFileLoader(dir, flist=test_list, label=label)
    num_files = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sz,
        num_workers=workers,
        shuffle=False,
        collate_fn=alignCollate(imgW, imgH)
    )

    image = Variable(image)

    # for p in crnn.parameters():
    #     p.requires_grad = False
    model.eval()
    print('Start predict')
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    print('Number of samples', num_files)
    begin = time.time()
    with torch.no_grad():
        for i in range(max_iter):
            data = val_iter.next()
            cpu_images, cpu_texts, img_paths = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            preds = model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            # print('    ', raw_pred)
            #print(' =>', sim_pred)
            #print(sim_pred)
            #print(img_paths)
            print(cpu_texts[0])
            if debug:
                inv_tensor = inv_normalize(cpu_images[0])
                cv_img = inv_tensor.permute(1, 2, 0).numpy()
                cv_img_convert = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                cv2.imshow('image data', cv_img_convert)
                cv2.waitKey(0)
    end = time.time()
    processing_time = end - begin
    print('Processing time:', processing_time)
    print('Speed:', num_files / processing_time, 'fps')


if __name__ == "__main__":
    list_dir = get_list_dir_in_folder(img_dir)
    if len(list_dir)>0:
        for subdir in list_dir:
            predict(os.path.join(img_dir, subdir), batch_size)
    else:
        predict(img_dir, batch_size)
