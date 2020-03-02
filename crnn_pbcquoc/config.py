#common
imgW = 1024
imgH = 64
alphabet_path = 'data/char_246'

#train
train_dir='/data/train_data_29k_29Feb'
pretrained='outputs/train_2020-02-29_11-31/AICR_ocr_dataset_mod_22.pth'
pretrained=''
gpu_train = '1'  #'0,1' or None
base_lr = 0.0005
max_epoches = 200
workers_train = 8
batch_size = 64
ckpt_prefix = 'AICR_finetune_new_data_128'

#test
test_dir='/home/aicr/cuongnd/text_recognition/data/handwriting/hokhau'
test_dir='/data/train_data_29k_29Feb/cleaned_data_merge_fixed/AICR_test2'
#pretrained_test='outputs/train_2020-02-20_09-03_finetune_cinamon_input32/AICR_pretrained_59.pth'  #appr 1
#pretrained_test='outputs/train_2020-02-20_16-58_train_cinamon_data/AICR_pretrained_74.pth'  #appr 2
#pretrained_test='outputs/train_2020-02-23_16-21_finetune_cinamon_input64/AICR_pretrained_53.pth'  #appr 3
#pretrained_test='outputs/train_2020-02-24_20-41_finetune_cinamon_input64_augmented/AICR_pretrained_61.pth'  #appr 4 AICR_pretrained_7
pretrained_test='outputs/train_2020-03-01_22-13_finetune_new_data2/AICR_finetune_new_data_44_loss_7.266_cer_0.028.pth'  #to test NLP
#pretrained_test='outputs/train_2020-03-02_10-05/AICR_finetune_new_data_128_1.pth'
label = True
test_list=''

gpu_test = '1'
gpu_test = None
workers_test=16
batch_size_test= 16
debug = False
if debug:
    batch_size_test=1

