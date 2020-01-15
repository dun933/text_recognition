import os, codecs, cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.utils import crop_from_img_square
from keras.models import Model, load_model
from datetime import datetime
from config.config_manager import ConfigManager
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.preprocessing import image

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat

configfile = 'config/classifier_config.ini'
configmanager = ConfigManager(configfile)
img_dir='/home/advlab/data/test_vn/test_image_vn/image_fix'
result_detector_dir='/data/CuongND/aicr_dssd_train/outputs/predict_2019-10-11_11-29_9674_raw'
#result_detector_dir='/data/CuongND/aicr_dssd_train/outputs/predict_2019-10-22_05-19_9762'
GT_dir='/home/advlab/data/test_vn/test_image_vn/v5/ground_truth'

training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir= 'outputs/predict_from_detector_'+ training_time
file_list = [
    '20190731_144554',
    '20190731_144540',
    '190715070245517_8478000669_pod',
    '190715070249216_8477872491_pod',
    '190715070317353_8479413342_pod'
]
input_size=56
thres_detector=0.51
thres_classifier1=0.3
work_dir='/data/CuongND/aicr_vn/aicr_classification_train/'
model_classifier1_path = work_dir+'outputs/train_classifier1_2019-10-22_14-32/weights_04-0.9842.hdf5' #9842
model_classifier2_A_path =  work_dir+'outputs/train_classifier2_A_2019-10-22_11-02/weights_07-0.27.hdf5' #9382 #9680
model_classifier2_alp_path =  work_dir+'outputs/train_classifier2_alp_2019-10-22_11-52/weights_08-0.99.hdf5' #9930
model_classifier2_E_path = work_dir+'outputs/train_classifier2_E_2019-10-22_11-51/weights_12-0.92.hdf5' #9240 #9430
model_classifier2_I_path =  work_dir+'outputs/train_classifier2_I_2019-10-22_11-53/weights_06-0.92.hdf5' #9182 #9349
model_classifier2_num_path =  work_dir+'outputs/train_classifier2_num_2019-10-22_12-49/weights_07-0.9857.hdf5' #9857
model_classifier2_O_path =  work_dir+'outputs/train_classifier2_O_2019-10-22_12-49/weights_07-0.8889.hdf5' #8889 #9674
model_classifier2_sym_path =  work_dir+'outputs/train_classifier2_sym_2019-10-16_16-31/weights_04-0.21.hdf5' #9520 #9280
model_classifier2_U_path =  work_dir+'outputs/train_classifier2_U_2019-10-22_12-51/weights_02-0.9691.hdf5' #9691
model_classifier2_Y_path =  work_dir+'outputs/train_classifier2_Y_2019-10-22_14-31/weights_03-0.9896.hdf5' #9896

model_classifier2 ={}
class_name=['A','E','I','O','U','Y','alp','num','sym']
class_list={} #total 227-5470  -->201 after ignore lower char
class_list['A'] = configmanager.class_list_A # 37-444
class_list['E'] = configmanager.class_list_E #24-270
class_list['O'] = configmanager.class_list_O #45-836 --> 26 after ignore lower char
class_list['I'] = configmanager.class_list_I #30-970
class_list['U'] = configmanager.class_list_U #24-198
class_list['Y'] =  configmanager.class_list_Y  #16-195 --> 13 after ignore lower char
class_list['num'] = configmanager.class_list_num #14-849 --> 13 after ignore lower char
class_list['sym'] = configmanager.class_list_sym #20-255
class_list['alp'] = configmanager.class_list_alp #17-1453 --> 14 after ignore lower char

def preprocess_img(img_data):
#     test_img= image.array_to_img(img_data)
# test_img=image.
#     test_img = image.img_to_array(test_img)
#
#     test_img = test_img * (1. / 255.)
#     x = test_img[np.newaxis, :]
    img = cv2.resize(img_data, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    img = img * (1. / 255.)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=2)
    x = img[np.newaxis, :]
    return x

def pred_from_detector(conf_thres_detector=thres_detector, conf_thres_classifier1=thres_classifier1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print ('Load classifier1',model_classifier1_path)
    model_classifier1 = load_model(model_classifier1_path)
    print ('Load classifier2_A')
    model_classifier2['A'] = load_model(model_classifier2_A_path)
    print ('Load classifier2_E')
    model_classifier2['E'] = load_model(model_classifier2_E_path)
    print ('Load classifier2_O')
    model_classifier2['O'] = load_model(model_classifier2_O_path)
    print ('Load classifier2_I')
    model_classifier2['I'] = load_model(model_classifier2_I_path)
    print ('Load classifier2_U')
    model_classifier2['U'] = load_model(model_classifier2_U_path)
    print ('Load classifier2_Y')
    model_classifier2['Y'] = load_model(model_classifier2_Y_path)
    print ('Load classifier2_num')
    model_classifier2['num'] = load_model(model_classifier2_num_path)
    print ('Load classifier2_sym')
    model_classifier2['sym'] = load_model(model_classifier2_sym_path)
    print ('Load classifier2_alp')
    model_classifier2['alp'] = load_model(model_classifier2_alp_path)

    for file_name in file_list:
        print ('Begin predict file',file_name)
        ori_img_path = os.path.join(img_dir, file_name + '.png')
        GT_file = os.path.join(result_detector_dir, file_name + '.txt')
        if os.path.exists(os.path.join(img_dir, file_name + '.jpg')):
            ori_img_path = os.path.join(img_dir, file_name + '.jpg')
        fh1 = codecs.open(GT_file, "r", encoding='utf-8-sig')
        img = cv2.imread(ori_img_path,  cv2.IMREAD_GRAYSCALE)
        result=''
        for idx, line in enumerate(fh1):
            data=(line.replace("\n", "")).split(" ")
            if(float(data[1])<conf_thres_detector): #ignore boudingbox with low confident
                continue
            xmin = int(data[2])
            ymin = int(data[3])
            width = int(data[4])
            height = int(data[5])
            coords = [xmin, ymin, xmin+width, ymin+height]
            roi = crop_from_img_square(img, coords)
            char_data=preprocess_img(roi)
            result_classifier1 = model_classifier1.predict(char_data)
            max_idx_classifier1 = np.argmax(result_classifier1, axis=1)
            max_val_classifier1 = np.amax(result_classifier1)
            if(max_val_classifier1<conf_thres_classifier1): #ignore roi that has low confident
                continue
            group_name=class_name[max_idx_classifier1[0]]
            result_classifier2 = model_classifier2[group_name].predict(char_data)
            max_idx_classifier2 = np.argmax(result_classifier2, axis=1)
            max_val_classifier2 = np.amax(result_classifier2)
            final_pred_char = class_list[group_name][max_idx_classifier2[0]]
            result_line=final_pred_char + ' ' + data[2] + ' ' + data[3]+ ' ' + data[4]+ ' ' + data[5]
            result+=result_line+'\n'
            print (result_line)

        with open(os.path.join(output_dir, file_name+'.txt'), "w") as f:
            f.write(result)

    #calculate accuracy
    get_F1_accuracy(GT_dir, output_dir,list_file=file_list)

def getBoundingBoxes(directory, list_file, isGT, allBoundingBoxes=None, imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    # Read ground truths
    # os.chdir(directory)
    for file_name in list_file:
        nameOfImage = file_name
        fh1 = codecs.open(os.path.join(directory, file_name+'.txt'), "r", encoding='utf8')
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            # print(line)
            if isGT:
                if splitLine[0] != '':
                    idClass=splitLine[0]
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])

                    bb = BoundingBox(
                        nameOfImage,
                        idClass, x, y, w, h,
                        'abs',
                        imgSize,
                        BBType.GroundTruth,
                        format=BBFormat.XYWH)
            else:  # detection
                if splitLine[0] != '':
                    idClass=splitLine[0]
                    # confidence = float(splitLine[1])
                    # x = float(splitLine[2])
                    # y = float(splitLine[3])
                    # w = float(splitLine[4])
                    # h = float(splitLine[5])
                    confidence=0.9
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])

                    bb = BoundingBox(
                        nameOfImage,
                        idClass, x, y, w, h,
                        'abs',
                        imgSize,
                        BBType.Detected,
                        confidence,
                        format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

def get_F1_accuracy(GT_dir, result_save_dir, list_file):
    # Get groundtruth boxes
    allBoundingBoxes = getBoundingBoxes(GT_dir, list_file, True)
    # Get detected boxes
    allBoundingBoxes = getBoundingBoxes(result_save_dir, list_file, False, allBoundingBoxes)

    evaluator = Evaluator()
    result=''

    tf = evaluator.GetTruePositive(
        allBoundingBoxes)  # Object containing all bounding boxes (ground truths and detections)

    TP_gt_vn = tf['TP_gt_vn']
    FN_gt_vn = tf['FN_gt_vn']
    TP_gt_alp = tf['TP_gt_alp']
    FN_gt_alp = tf['FN_gt_alp']
    TP_gt_sym = tf['TP_gt_sym']
    FN_gt_sym = tf['FN_gt_sym']
    TP_gt_num = tf['TP_gt_num']
    FN_gt_num = tf['FN_gt_num']

    acc_vn=round(TP_gt_vn / (TP_gt_vn + FN_gt_vn), 4)
    acc_alp=round(TP_gt_alp / (TP_gt_alp + FN_gt_alp), 4)
    acc_sym=round(TP_gt_sym / (TP_gt_sym + FN_gt_sym), 4)
    acc_num=round(TP_gt_num / (TP_gt_num + FN_gt_num), 4)
    total_TP=TP_gt_vn + TP_gt_alp + TP_gt_sym + TP_gt_num
    total_FN=FN_gt_vn + FN_gt_alp + FN_gt_sym + FN_gt_num
    F1=round(total_TP / (total_TP + total_FN), 4)

    result +='class vn: ' + str(acc_vn) + ' TP: ' + str(TP_gt_vn) + ' FP: ' + str(FN_gt_vn)+'\n'
    result +='class alp: ' + str(acc_alp) + ' TP: ' + str(TP_gt_alp) + ' FP: ' + str(FN_gt_alp)+'\n'
    result +='class sym: ' + str(acc_sym) + ' TP: ' + str(TP_gt_sym) + ' FP: ' + str(FN_gt_sym)+'\n'
    result +='class num: ' + str(acc_num) + ' TP: ' + str(TP_gt_num) + ' FP: ' + str(FN_gt_num)+'\n'
    #result +='Final F1: '+str(F1)+' total TP: '+str(total_TP)+' total FN: '+str(total_FN)+' total samples: '+str(total_TP+total_FN)+'\n'

    #get F1 score
    detections = evaluator.GetF1ScoreMetrics(allBoundingBoxes)
    tp = detections['TP']
    fp = detections['FP']
    fn = detections['FN']

    result+='tp: ' +str(tp)+ ', fp: ' +str(fp)+ ', fn: ' +str(fn)+'\n'
    precision = round(tp / (tp + fp),4)
    recall = round(tp / (tp + fn),4)
    result+='precision: '+str(precision)+', recall: '+str(recall)+'\n'
    if precision > 0 and recall > 0:
        f1 = round((2 * precision * recall) / (precision + recall),4)
    else:
        f1 = 0
    result+='F1 score: '+str(f1)
    print(result)
    with open(os.path.join(result_save_dir, 'result.txt'), "w") as f:
        f.write(result)

if __name__== "__main__":
    pred_from_detector()
    #result_3stages_dir = '/data/CuongND/aicr_vn/aicr_classification_train/outputs/predict_from_detector_2019-10-21_04-52_8931'
    #get_F1_accuracy(GT_dir, result_3stages_dir,list_file=file_list)