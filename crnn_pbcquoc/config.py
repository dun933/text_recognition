#common
imgW = 512
imgH = 64


#train
train_dir='/home/duycuong/PycharmProjects/dataset/ocr_dataset'
#pretrained='outputs/train_2020-02-20_18-00/AICR_pretrained_13.pth'
pretrained=''
gpu_train = '0'  #'0,1' or None
base_lr = 0.0005
max_epoches = 100
alphabet_name = 'data/char_246'
workers_train=8
batch_size=64
ckpt_prefix='AICR_crnn'


#test
test_dir='/home/duycuong/PycharmProjects/dataset/cinnamon_data'
pretrained_test='outputs/pretrain_ocr_dataset/AICR_pretrained_30.pth'

#gpu_test = '0'
gpu_test = None
workers_test=4
batch_size_test=64

