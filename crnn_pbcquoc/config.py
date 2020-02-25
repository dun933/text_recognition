#common
imgW = 1024
imgH = 64
alphabet_path = 'data/char_246'


#train
train_dir='/data/dataset/cinnamon_data'
#pretrained='outputs/train_2020-02-21_13-44_train_ocr_dataset/AICR_pretrained_30.pth'
pretrained=''
gpu_train = '0'  #'0,1' or None
base_lr = 0.001
max_epoches = 200
workers_train = 4
batch_size = 64
ckpt_prefix = 'AICR_pretrained'

#test
test_dir='/data/dataset/cinnamon_data/0916_DataSamples'
test_dir='../data/handwriting/form_crop2'
pretrained_test='outputs/train_2020-02-20_09-03_finetune_cinamon_input32/AICR_pretrained_59.pth'  #appr 1
#pretrained_test='outputs/train_2020-02-20_16-58_train_cinamon_data/AICR_pretrained_74.pth'  #appr 2
#pretrained_test='outputs/train_2020-02-23_16-21_finetune_cinamon_input64/AICR_pretrained_53.pth'  #appr 3
pretrained_test='outputs/train_2020-02-24_20-41_finetune_cinamon_input64_augmented/AICR_pretrained_61.pth'  #appr 4
label = False
test_list=''

gpu_test = '0'
#gpu_test = None
workers_test=16
batch_size_test=16
debug = True
if debug:
    batch_size_test=1

