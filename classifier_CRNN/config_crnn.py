#common
imgW = 1024
imgH = 64
alphabet_path = 'data/char_246'

#train
train_dir='/data/train_data_29k_29Feb_update30Mar'
pretrained='outputs/train_2020-02-29_11-31_train_ocr_dataset_mod_64/AICR_ocr_dataset_mod_32_loss_3.217_cer_0.0278.pth'
#pretrained='outputs/train_2020-02-21_13-44_train_ocr_dataset/AICR_pretrained_30_Test_3.808_cer_0.0269.pth'
#pretrained=''
#pretrained='ckpt/AICR_finetune_new_data_44_loss_7.266_cer_0.028.pth'
gpu_train = '0'  #'0,1' or None
base_lr = 0.0003
max_epoches = 500
workers_train = 8
batch_size = 64
ckpt_prefix = 'AICR_SDV_30Mar_No_update_hw'

#test
test_dir='/home/aicr/cuongnd/text_recognition/data/handwriting/hokhau'
test_dir='data'
pretrained_test='outputs/train_2020-03-30_18-38/AICR_SDV_30Mar_No_update_hw_300_loss_1.25_cer_0.0076.pth'
pretrained_test='ckpt/AICR_SDV_30Mar_300_loss_1.25_cer_0.0076.pth'
#for testing SDV printing character cases
label = False
test_list=''

gpu_test = '0'
#gpu_test = None
workers_test=8
batch_size_test= 1
debug = False
if debug:
    batch_size_test=1
