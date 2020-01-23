import codecs
import os, sys
import warnings

from config.config_manager import ConfigManager

warnings.filterwarnings("ignore")
from datetime import datetime

from predict_core import SSD_Predict
from utils.split_image import split_image_to_objs
from utils.run_util import drawAnnotation, calculateIOU
from utils.image_processing_util import zoomScaleFinder

import tensorflow as tf
from keras import backend as k

from metric.BoundingBox import BoundingBox
from metric.BoundingBoxes import BoundingBoxes
from metric.Evaluator import *
from metric.utils import BBFormat

configfile = 'config/vietnamese_config.ini' #
configmanager = ConfigManager(configfile)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configmanager.infer_gpu_num
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

img_shape = configmanager.img_shape
classes = configmanager.classes
conf_thres = 0.35
iou_threshold = 0.35
nms_thres = 0.45
max_size = 2000

save_anno_image = True
img_font_idle_size = configmanager.img_font_idle_size
img_font_idle_size2 = configmanager.img_font_idle_size2
NMS_ALGORITHM = configmanager.nms_algorithm

split_overap_size = configmanager.split_overap_size
# iou_threshold = configmanager.iou_threshold
weight_path = configmanager.ssd_weight_path
weight_name = os.path.basename(weight_path)
result_save_dir = 'outputs/predict_' + training_time + '_' + weight_name.replace('.hdf5','')

img_dir = '/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data/Cello_Vietnamese_TestSet/'
#img_dir='/data/CuongND/SDSC2_IMG'
GT_dir = '/home/duycuong/PycharmProjects/research_py3/text_recognition/data/Cello/v7/ground_truth'
#from data_processing import  get_list_file_in_folder
#file_list=get_list_file_in_folder(img_dir)
#file_list = [x.replace('.png','').replace('.jpg','') for x in file_list]
file_list = [
    '20190731_144554',
    '20190731_144540',
    '190715070245517_8478000669_pod',
    '190715070249216_8477872491_pod',
    '190715070317353_8479413342_pod'
]

class writer:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, text):
        for w in self.writers:
            w.write(text)
    def flush(self):
        pass

def save_predict_result(coord_list, save_result_filename='test.txt', zoom_rate=1):
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


def predict(Metric=None):
    print('Begin predict with checkpoint:', weight_path, '\nSave result to:', result_save_dir)
    print('Img font min size',img_font_idle_size,', Img font max size',img_font_idle_size2)
    ssd_predict = SSD_Predict(classes=classes,
                              weight_path=weight_path,
                              input_shape=img_shape,
                              nms_thresh=nms_thres,
                              NMS_ALGORITHM=NMS_ALGORITHM)
    for file_name in file_list:
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
        while min(w_, h_) < max_size:
            z__ = 2
            z_ = z_ * 2
            ori_img_gray = cv2.resize(ori_img_gray, (int(w_ * z__), int(h_ * z__)), interpolation=cv2.INTER_LANCZOS4)
            h_, w_ = ori_img_gray.shape
            print('Shape of resize image (h,w):', ori_img_gray.shape, 'zoom_ratio', z_)

        # 문서 내 대다수의 폰트 크기 추정
        img_font_regular_size, avg_character_height = zoomScaleFinder(ori_img_gray, h=float(max_size), AVG_RATIO=0.8,
                                                                      DEBUG=False)
        # Zoom In/Out 비율 획득

        zoom_ratio1 = round(img_font_idle_size / img_font_regular_size, 2)
        #zoom_ratio1 = 0.58
        print('img_font_regular_size : %.2f' % img_font_regular_size + ' , zoom_ratio : %.2f' % zoom_ratio1)

        # 이미지 리사이즈
        ori_img_gray_resize = cv2.resize(ori_img_gray, None, fx=zoom_ratio1, fy=zoom_ratio1,
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
        ori_img_gray_resize = cv2.resize(ori_img_gray, None, fx=zoom_ratio2, fy=zoom_ratio2, interpolation=cv2.INTER_CUBIC)

        # split image for object detection
        img_obj_list2, img_coord_list2 = split_image_to_objs(imgage_obj=ori_img_gray_resize,
                                                             img_shape=img_shape, overap_size=split_overap_size,
                                                             zoom_ratio=zoom_ratio2)

        img_obj_list = img_obj_list + img_obj_list2
        img_coord_list = img_coord_list + img_coord_list2

        ssd_predict_result_list = ssd_predict.predict_from_obj_list(img_obj_list, img_coord_list, conf_thres=conf_thres)
        ssd_predict_result_list = sorted(ssd_predict_result_list)

        print('ssd predict result list count: ', str(len(ssd_predict_result_list)))

        # Calculated IOU and applied
        applied_iou_list = calculateIOU(ssd_predict_result_list, iou_threshold)

        print('appied iou list count: ', str(len(applied_iou_list)))

        result_save_path = os.path.join(result_save_dir,
                                        file_name + '_annot_iou_{}_nms_{}.jpg'.format(iou_threshold, nms_thres))
        save_predict_result(applied_iou_list, os.path.join(result_save_dir, file_name + '.txt'), z_)
        if save_anno_image:
            drawAnnotation(applied_iou_list, ori_img_gray, show_conf=False, save_file_name=result_save_path)
        if Metric is not None:
            Metric([file_name], type=file_name)
    # calculte F1-Score
    print()
    if Metric is not None:
        Metric(file_list, type='all')

def getBoundingBoxes(directory, list_file, isGT, allBoundingBoxes=None, allClasses=None, imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
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
                    classID = "all"
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])

                    bb = BoundingBox(
                        nameOfImage,
                        classID, x, y, w, h,
                        'abs',
                        imgSize,
                        BBType.GroundTruth,
                        format=BBFormat.XYWH)
            else:  # detection
                if splitLine[0] != '':
                    classID = "all"
                    confidence = float(splitLine[1])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])

                    bb = BoundingBox(
                        nameOfImage,
                        classID, x, y, w, h,
                        'abs',
                        imgSize,
                        BBType.Detected,
                        confidence,
                        format=BBFormat.XYWH)
            if isGT:
                allBoundingBoxes.addBoundingBox(bb)
            if not isGT and (confidence >= conf_thres) and h > 0:
                allBoundingBoxes.addBoundingBox(bb)
            if classID not in allClasses:
                allClasses.append(classID)
        fh1.close()
    return allBoundingBoxes, allClasses

def get_F1Score(list_file, type=''):
    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(GT_dir, list_file, True)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(result_save_dir, list_file, False, allBoundingBoxes, allClasses)
    allClasses.sort()
    evaluator = Evaluator()

    tf = evaluator.GetDetectorTruePositive(allBoundingBoxes)
    TP_gt = tf['TP_gt']
    FN_gt = tf['FN_gt']

    detections = evaluator.GetPascalVOCMetrics(allBoundingBoxes,  IOUThreshold=iou_threshold)
    # each detection is a class
    # for metricsPerClass in detections:
    metricsOneClass = detections[0]
    # Get metric values per each class
    cl = metricsOneClass['class']
    ap = metricsOneClass['AP']
    precision = metricsOneClass['precision']
    recall = metricsOneClass['recall']
    totalPositives = metricsOneClass['total positives']
    total_TP = metricsOneClass['total TP']
    total_FP = metricsOneClass['total FP']

    print('Result of ' + type + ':')
    print('F1 score: ' + str(round(TP_gt / (TP_gt + FN_gt),4)) + ' | TP (from GT): ' + str(TP_gt) + '| FN: ' + str(FN_gt))
    print('AP: Recall: ' + str(round(total_TP / (total_FP + total_TP),4)) + ' | TP: ' + str(total_TP) + ' | FP: ' + str(
        total_FP))

if __name__ == "__main__":
    saved = sys.stdout
    if not os.path.isdir(result_save_dir):
        os.mkdir(result_save_dir)
    log_file = os.path.join(result_save_dir, "predict.log")
    f = open(log_file, 'w')
    sys.stdout = writer(sys.stdout, f)
    print('Confident thres: ' + str(conf_thres) + ' IoU: ' + str(iou_threshold) + ' NMS: ' + str(nms_thres))
    predict()
    #predict(Metric=get_F1Score)
    sys.stdout = saved
    f.close()
