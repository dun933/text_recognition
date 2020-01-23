import codecs, json
import os, sys
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
import time
import tensorflow as tf
from keras import backend as k
from keras.models import Model, load_model
import cv2
import numpy as np

# import for detector
from aicr_dssd_train.predict_core import SSD_Predict
from aicr_dssd_train.utils.split_image import split_image_to_objs
from aicr_dssd_train.utils.run_util import drawAnnotation, calculateIOU, calculateIOUCNX
from aicr_dssd_train.utils.image_processing_util import zoomScaleFinder

from aicr_classification_train.lib.Evaluator import *

# import for classifier
from aicr_classification_train.utils.utils import crop_from_img_square, crop_from_img_rectangle
from data_processing import draw_annot_by_text
from aicr_classification_test.aicr.document.lineup import line_out, line_out2, line_out3


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

# detector
img_shape = (320, 320, 3)
classes = ["vietnam"]
thres_detector = 0.35
iou_threshold = 0.35
nms_thres = 0.45
max_size = 2000

#filter all box which is out of range
min_h = 5
min_w = 3
max_h = 320
max_w = 320

save_anno_image = True
img_font_idle_size = 28
img_font_idle_size2 = 40
NMS_ALGORITHM = "NMS"
measure_speed_only = False
parallel = True
word_compose = True
split_overap_size = 50
weight_path = "model/detector.hdf5"
weight_name = os.path.basename(weight_path)
result_save_dir = 'outputs/predict_2stages_' + pred_time

# img_dir = '/home/advlab/data/test_vn/Cello_data/test_image_vn/image_fix/'
# GT_dir = '/home/advlab/data/test_vn/Cello_data/test_image_vn/v7/ground_truth/'
img_dir = '/home/advlab/data/test_vn/Eval_data/imgs/'
GT_dir = '/home/advlab/data/test_vn/Eval_data/anno/v1/'
file_list=get_list_file_in_folder(img_dir)
file_list = [x.replace('.png','').replace('.jpg','') for x in file_list]
# classifier
input_shape = (36, 72, 1)
thres_classifier12 = 0.35
wh_ratio = 2
rect_mode = 1
model_classifier12_path = 'model/classifier_Cello.hdf5'
with open('aicr_classification_train/config/classmap_2stages_228.json', 'r') as f:
    loaded_json = json.load(f)
class_list = []
for cls in loaded_json.keys():
    class_list.append(cls)

classes_vn = "ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨ" \
             "ÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờùừỳ" \
             "ảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
class_list_vn = [x for x in classes_vn]

classes_symbol = '*:,@$.-(#%\'\")/~!^&_+={}[]\;<>?※”'
class_list_symbol = [x for x in classes_symbol]

classes_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
class_list_alphabet = [x for x in classes_alphabet]

classes_number = '0123456789'
class_list_number = [x for x in classes_number]


class writer:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

    def flush(self):
        pass

def save_detector_result(coord_list, save_result_filename='test.txt', zoom_rate=1):
    result = ''
    for c in coord_list:
        class_nm = c.class_nm
        conf = round(c.confidence, 2)
        c = c.getAbsolute_coord()
        x_min = int(round(c[0] / zoom_rate, 2))
        y_min = int(round(c[1] / zoom_rate, 2))
        width = int(round((c[2] - c[0]) / zoom_rate, 2))
        height = int(round((c[3] - c[1]) / zoom_rate, 2))
        result_line = class_nm + ' ' + str(conf) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(width) + ' ' + str(
            height)
        # print(result_line)
        result += result_line + '\n'
    with open(save_result_filename, "w") as f:
        f.write(result)


def preprocess_img(img_data):
    img = cv2.resize(img_data, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite('roi.jpg',img)
    img = img * (1. / 255.)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=2)
    x = img[np.newaxis, :]
    return x


def getBoundingBoxes(directory, list_file, isGT, allBoundingBoxes=None, imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    # Read ground truths
    # os.chdir(directory)
    for file_name in list_file:
        nameOfImage = file_name
        fh1 = codecs.open(os.path.join(directory, file_name + '.txt'), "r", encoding='utf8')
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            # print(line)
            if isGT:
                if splitLine[0] != '':
                    idClass = splitLine[0]
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
                    idClass = splitLine[0]
                    if idClass == 'dot':  # change . folder to dot folder
                        idClass = '.'
                    if idClass == 'slash':
                        idClass = '/'
                    confidence = float(splitLine[1])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
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
    print('\nBegin calculate F1-score')
    result = 'Thres_detector: ' + str(thres_detector) + ' IoU: ' + str(iou_threshold) + ' NMS: ' + str(nms_thres) + '\n'
    result += 'Thres_classifier12: ' + str(thres_classifier12) + '\n'

    tf = evaluator.GetTruePositive(allBoundingBoxes)

    TP_gt_vn = tf['TP_gt_vn']
    FN_gt_vn = tf['FN_gt_vn']
    TP_gt_alp = tf['TP_gt_alp']
    FN_gt_alp = tf['FN_gt_alp']
    TP_gt_sym = tf['TP_gt_sym']
    FN_gt_sym = tf['FN_gt_sym']
    TP_gt_num = tf['TP_gt_num']
    FN_gt_num = tf['FN_gt_num']

    acc_vn = round(TP_gt_vn / (TP_gt_vn + FN_gt_vn), 4)
    acc_alp = round(TP_gt_alp / (TP_gt_alp + FN_gt_alp), 4)
    acc_sym = round(TP_gt_sym / (TP_gt_sym + FN_gt_sym), 4)
    acc_num = round(TP_gt_num / (TP_gt_num + FN_gt_num), 4)
    total_TP = TP_gt_vn + TP_gt_alp + TP_gt_sym + TP_gt_num
    total_FN = FN_gt_vn + FN_gt_alp + FN_gt_sym + FN_gt_num
    F1 = round(total_TP / (total_TP + total_FN), 4)

    result += 'class Vietnam: ' + str(acc_vn) + ' TP: ' + str(TP_gt_vn) + ' FP: ' + str(FN_gt_vn) + '\n'
    result += 'class English: ' + str(acc_alp) + ' TP: ' + str(TP_gt_alp) + ' FP: ' + str(FN_gt_alp) + '\n'
    result += 'class Symbol: ' + str(acc_sym) + ' TP: ' + str(TP_gt_sym) + ' FP: ' + str(FN_gt_sym) + '\n'
    result += 'class Number: ' + str(acc_num) + ' TP: ' + str(TP_gt_num) + ' FP: ' + str(FN_gt_num) + '\n'
    # result +='Final F1: '+str(F1)+' total TP: '+str(total_TP)+' total FN: '+str(total_FN)+' total samples: '+str(total_TP+total_FN)+'\n'

    # get F1 score
    detections = evaluator.GetF1ScoreMetrics(allBoundingBoxes)
    tp = detections['TP']
    fp = detections['FP']
    fn = detections['FN']

    result += 'TP: ' + str(tp) + ', FP: ' + str(fp) + ', FN: ' + str(fn) + '\n'
    precision = round(tp / (tp + fp), 4)
    recall = round(tp / (tp + fn), 4)
    result += 'Precision: ' + str(precision) + ', Recall: ' + str(recall) + '\n'
    if precision > 0 and recall > 0:
        f1 = round((2 * precision * recall) / (precision + recall), 4)
    else:
        f1 = 0
    result += 'F1 score: ' + str(f1)
    print(result)
    with open(os.path.join(result_save_dir, 'result.txt'), "w") as f:
        f.write(result)


def predict():
    print('Begin predict. Save result to:', result_save_dir)
    begin_init = time.time()
    print('Load detector', weight_path)
    model_detector = SSD_Predict(classes=classes,
                                 weight_path=weight_path,
                                 input_shape=img_shape,
                                 nms_thresh=nms_thres,
                                 NMS_ALGORITHM=NMS_ALGORITHM)
    print('Thres_detector: ' + str(thres_detector) + ' IoU: ' + str(iou_threshold) + ' NMS: ' + str(nms_thres))
    print('Load classifier12', model_classifier12_path)
    model_classifier12 = load_model(model_classifier12_path)
    end_init = time.time()
    print('Init model time:', round((end_init - begin_init), 4), 'seconds')
    total_char = 0
    begin = time.time()
    for file_name in file_list:
        time_detector_begin = time.time()
        ori_img_path = os.path.join(img_dir, file_name + '.png')
        if os.path.exists(os.path.join(img_dir, file_name + '.jpg')):
            ori_img_path = os.path.join(img_dir, file_name + '.jpg')
        # load original image with grayscale
        ori_img = cv2.imread(ori_img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        # 이미지 Grayscale로 변환
        if len(ori_img.shape) > 2:
            ori_img_gray = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            ori_img_gray = ori_img.copy()
        print('\nTest image:', ori_img_path, ', original shape(h,w):', ori_img_gray.shape)

        h_, w_ = ori_img_gray.shape
        z_ = 1
        ori_img_gray_mod = ori_img_gray.copy()
        while min(w_, h_) < max_size:
            z__ = 2
            z_ = z_ * 2
            ori_img_gray_mod = cv2.resize(ori_img_gray_mod, (int(w_ * z__), int(h_ * z__)),
                                          interpolation=cv2.INTER_LANCZOS4)
            h_, w_ = ori_img_gray_mod.shape
            print('Shape of resize image (h,w):', ori_img_gray_mod.shape, 'zoom_ratio', z_)

        # 문서 내 대다수의 폰트 크기 추정
        img_font_regular_size, avg_character_height = zoomScaleFinder(ori_img_gray_mod, h=float(max_size),
                                                                      AVG_RATIO=0.8,
                                                                      DEBUG=False)
        # Zoom In/Out 비율 획득

        zoom_ratio1 = round(img_font_idle_size / img_font_regular_size, 2)
        print('img_font_regular_size : %.2f' % img_font_regular_size + ' , zoom_ratio : %.2f' % zoom_ratio1)

        # 이미지 리사이즈
        ori_img_gray_resize = cv2.resize(ori_img_gray_mod, None, fx=zoom_ratio1, fy=zoom_ratio1,
                                         interpolation=cv2.INTER_CUBIC)

        # split image for object detection
        img_obj_list, img_coord_list = split_image_to_objs(imgage_obj=ori_img_gray_resize,
                                                           img_shape=img_shape, overap_size=split_overap_size,
                                                           zoom_ratio=zoom_ratio1)

        # Zoom In/Out 비율 획득
        zoom_ratio2 = round(img_font_idle_size2 / img_font_regular_size, 2)
        # zoom_ratio2 = 0.4
        print('img_font_regular_size : %.2f' % img_font_regular_size + ' , zoom_ratio : %.2f' % zoom_ratio2)

        # 이미지 리사이즈
        ori_img_gray_resize = cv2.resize(ori_img_gray_mod, None, fx=zoom_ratio2, fy=zoom_ratio2,
                                         interpolation=cv2.INTER_CUBIC)

        # split image for object detection
        img_obj_list2, img_coord_list2 = split_image_to_objs(imgage_obj=ori_img_gray_resize,
                                                             img_shape=img_shape, overap_size=split_overap_size,
                                                             zoom_ratio=zoom_ratio2)

        img_obj_list = img_obj_list + img_obj_list2
        img_coord_list = img_coord_list + img_coord_list2

        ssd_predict_result_list = model_detector.predict_from_obj_list(img_obj_list, img_coord_list,
                                                                       conf_thres=thres_detector)
        ssd_predict_result_list = sorted(ssd_predict_result_list)

        print('ssd predict result list count: ', str(len(ssd_predict_result_list)))

        # Calculated IOU and applied
        applied_iou_list = calculateIOUCNX(ssd_predict_result_list, iou_threshold)

        time_detector_end = time.time()
        print('Predict detector time:', round(time_detector_end - time_detector_begin, 4), 'seconds')

        result = ''
        if not measure_speed_only:  # save data
            save_detector_result(applied_iou_list, os.path.join(result_save_dir, file_name + '_detector.txt'), z_)

        # classifier
        print('Begin classify', len(applied_iou_list), 'char')
        total_char += len(applied_iou_list)
        image_list = []
        final_point_obj_list = []
        for bbox in applied_iou_list:
            conf = round(bbox.confidence, 2)
            if (conf < thres_detector):  # ignore boudingbox with low confident
                continue
            c = bbox.getAbsolute_coord()
            x_min = int(round(c[0] / z_, 2))
            y_min = int(round(c[1] / z_, 2))
            width = int(round((c[2] - c[0]) / z_, 2))
            height = int(round((c[3] - c[1]) / z_, 2))
            coords = [x_min, y_min, x_min + width, y_min + height]
            if (width < min_w) or (height < min_h) or (width > max_w) or (height > max_h):
                continue
            try:
                roi = crop_from_img_rectangle(ori_img_gray, coords, wh_ratio=wh_ratio, mode=rect_mode)
                char_data = preprocess_img(roi)
                image_list.append(char_data.copy())
                final_point_obj_list.append(bbox)
                if parallel is True:
                    continue
                result_classifier12 = model_classifier12.predict(char_data)
            except:
                print('Exception:', str(x_min) + ' ' + str(y_min) + ' ' + str(width) + ' ' + str(height))
                continue

            if not measure_speed_only:  # save data
                max_idx_classifier12 = np.argmax(result_classifier12, axis=1)
                max_val_classifier12 = round(np.amax(result_classifier12), 2)
                if (max_val_classifier12 < thres_classifier12):  # ignore roi that has low confident
                    continue
                final_pred_char = class_list[max_idx_classifier12[0]]
                if final_pred_char == 'dot':  # change . folder to dot folder
                    final_pred_char = '.'
                if final_pred_char == 'slash':
                    final_pred_char = '/'
                result_line = final_pred_char + ' ' + str(max_val_classifier12) + ' ' + str(x_min) + ' ' + str(
                    y_min) + ' ' + str(width) + ' ' + str(height)
                result += result_line + '\n'

        if parallel is True:
            batch_size = 128
            images = np.zeros((0, input_shape[0], input_shape[1], 1))
            for i, target in enumerate(image_list):
                images = np.vstack([images, target])
                if images.shape[0] == batch_size or i == len(image_list) - 1:
                    ret_classes = model_classifier12.predict(images, batch_size=batch_size)

            for i in range(len(image_list)):
                max_idx_classifier12 = np.argmax(ret_classes[i], axis=0)
                max_val_classifier12 = round(np.amax(ret_classes[i]), 2)
                if (max_val_classifier12 < thres_classifier12):  # ignore roi that has low confident
                    continue
                final_pred_char = class_list[max_idx_classifier12]
                if final_pred_char == 'dot':  # change . folder to dot folder
                    final_pred_char = '.'
                if final_pred_char == 'slash':
                    final_pred_char = '/'
                final_point_obj_list[i].setCharValue(final_pred_char)
                c = final_point_obj_list[i].getAbsolute_coord()
                x_min = int(round(c[0] / z_, 2))
                y_min = int(round(c[1] / z_, 2))
                width = int(round((c[2] - c[0]) / z_, 2))
                height = int(round((c[3] - c[1]) / z_, 2))
                if final_pred_char == ' ':
                    continue
                result_line = final_pred_char + ' ' + str(max_val_classifier12) + ' ' + str(x_min) + ' ' + str(
                    y_min) + ' ' + str(width) + ' ' + str(height)
                result += result_line + '\n'
            if word_compose:
                print("start compose to words...")
                word_list, str_line_result, line_result, line_result_word = line_out3(final_point_obj_list)
                decoded_result = ''
                for line in str_line_result:
                    list_word = []
                    for word in line_result_word[line]:
                        list_word.append(word[4])
                    str1 = ' '.join(list_word)
                    decoded_result += str1 + '\n'
                # print(decoded_result)
                with open(os.path.join(result_save_dir, file_name + '_decode_result.txt'), 'w', encoding='utf-8') as f:
                    f.write(decoded_result)
                print('------------End image--------------')

        if not measure_speed_only:
            with open(os.path.join(result_save_dir, file_name + '.txt'), "w") as f:
                f.write(result)
        time_classifier_end = time.time()
        print('Predict classifier time:', round(time_classifier_end - time_detector_end, 4), 'seconds')

    end = time.time()
    processing_time = end - begin
    print('Total predict time:', round(processing_time, 4), 'seconds')
    if (total_char > 0):
        print('Speed: 1000 char in', round(processing_time * 1000 / total_char, 4), 'seconds')

    if not measure_speed_only:
        if save_anno_image:
            print('\nVisualize result to file.')
            draw_annot_by_text(file_list, img_dir, result_save_dir,result_save_dir, isGT=False, draw_bbox=True)
        get_F1_accuracy(GT_dir, result_save_dir,list_file=file_list)


if __name__ == "__main__":
    saved = sys.stdout
    if not os.path.isdir(result_save_dir):
        os.mkdir(result_save_dir)
    log_file = os.path.join(result_save_dir, "predict.log")
    f = open(log_file, 'w')
    sys.stdout = writer(sys.stdout, f)
    predict()
    sys.stdout = saved
    f.close()
