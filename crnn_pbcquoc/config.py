#common
imgW = 1024
imgH = 64
alphabet_path = 'data/char_246'


#train
train_dir='/home/duycuong/PycharmProjects/dataset'
pretrained='outputs/train_2020-02-21_13-44_train_ocr_dataset/AICR_pretrained_30.pth'
#pretrained=''
gpu_train = '0'  #'0,1' or None
base_lr = 0.0005
max_epoches = 200
workers_train = 8
batch_size = 64
ckpt_prefix = 'AICR_finetune_29Feb_'

#test
test_dir='/data/dataset/cinnamon_data/1015_Private Test'
test_dir='/home/duycuong/PycharmProjects/dataset/cleaned_data_merge_fixed/AICR_test2'
#pretrained_test='outputs/train_2020-02-20_09-03_finetune_cinamon_input32/AICR_pretrained_59.pth'  #appr 1
#pretrained_test='outputs/train_2020-02-20_16-58_train_cinamon_data/AICR_pretrained_74.pth'  #appr 2
#pretrained_test='outputs/train_2020-02-23_16-21_finetune_cinamon_input64/AICR_pretrained_53.pth'  #appr 3
pretrained_test='outputs/train_2020-02-29_18-26/AICR_finetune_29Feb__30.pth'  #appr 4 AICR_pretrained_7
#pretrained_test='outputs/train_2020-02-24_20-41_finetune_cinamon_input64_augmented/AICR_pretrained_7.pth'  #to test NLP
label = False
test_list=''

gpu_test = '0'
#gpu_test = None
workers_test=16
batch_size_test= 1
debug = True
if debug:
    batch_size_test=1

