#common
imgW = 1024
imgH = 64
alphabet_path = 'data/char_246'


#train
train_dir='/data/dataset/ocr_dataset'
#pretrained='outputs/train_2020-02-20_18-00/AICR_pretrained_13.pth'
pretrained=''
gpu_train = '0'  #'0,1' or None
base_lr = 0.0005
max_epoches = 100
workers_train = 8
batch_size = 64
ckpt_prefix = 'AICR_crnn'

#test
test_dir='/data/dataset/cinnamon_data/0916_DataSamples'
test_dir='../data/handwriting/vib_form/crop'
pretrained_test='outputs/train_2020-02-23_16-21_finetune_cinamon_data/AICR_pretrained_48.pth'
label = False
test_list=''

gpu_test = '0'
#gpu_test = None
workers_test=16
batch_size_test=16
debug=True
if debug:
    batch_size_test=1

