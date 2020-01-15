import os, codecs, cv2
import warnings

from sqlalchemy.sql.functions import mode

warnings.filterwarnings("ignore")
import numpy as np
from utils.utils import crop_from_img_square, crop_from_img_by_margin
from keras.models import Model, load_model
from datetime import datetime
from config.config_manager_chinese import ConfigManager
from keras.utils.training_utils import multi_gpu_model
import json
from keras.models import model_from_json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


configfile = 'config/classifier_chinese_config.ini'
configmanager = ConfigManager(configfile)
os.environ["CUDA_VISIBLE_DEVICES"] = configmanager.gpu_num_classifier_infer

img_dir = '/data/chinese_images/SDSC2_IMG'
result_2nd_stage_dir = '/data/data_thang/detection_outputs/predict_2019-11-16_15-22/'

GT_dir = '/data/chinese_images/SDSC2_TXT'

training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = '/data/data_thang/classification_outputs/result_3nd_stage_' + training_time


thres_detector = 0.51
thres_classifier = 0.3


def load_imgfilepaths(dirpath):
    file_list = []
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            file_list.append(f[:-4])
            # file_list.append(os.path.join(root, f))

    return file_list


def preprocess_img(img_data):
    img = cv2.resize(img_data, (configmanager.input_size_classifier[0], configmanager.input_size_classifier[1]))
    img = img * (1. / 255.)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=2)
    x = img[np.newaxis, :]
    return x


def load_model_from_json(json_model_path):
    json_file = open(json_model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


def convert_multigpu_to_one(json_model_path, weight_path):
    model_ = load_model_from_json(json_model_path)
    model = multi_gpu_model(model_, gpus=2)
    model.load_weights(weight_path)

    return model.layers[-2]  # get single GPU model weights


def pred_from_2nd_stage(conf_thres_detector=thres_detector, conf_thres_classifier=thres_classifier):
    print('Load classifier', configmanager.model_path_classifier, ", ", configmanager.weight_path_classifier)
    model_classifier = convert_multigpu_to_one(configmanager.model_path_classifier, configmanager.weight_path_classifier)
    # model_classifier.save("/data/data_thang/detection_outputs/model_classifier_single.hdf5")
    file_list = load_imgfilepaths(img_dir)

    class_name_vs_id = json.loads(open(configmanager.class_map, 'r').read())
    # fh1 = codecs.open('/data/aicr_hanh/data_xau/class.txt', "w", encoding='utf-8-sig')
    # for key in class_name_vs_id:
    #     fh1.write(key)
    #     fh1.write('\n')
    # fh1.close()
    # exit(0)

    class_id_vs_name = {}
    for name in class_name_vs_id:
        class_id_vs_name[class_name_vs_id[name]] = name

    for file_name in file_list:
        print('Begin predict file', file_name)
        ori_img_path = os.path.join(img_dir, file_name + '.png')
        GT_file = os.path.join(result_2nd_stage_dir, file_name + '.txt')
        if os.path.exists(os.path.join(img_dir, file_name + '.jpg')):
            ori_img_path = os.path.join(img_dir, file_name + '.jpg')
        fh1 = codecs.open(GT_file, "r", encoding='utf-8-sig')
        img = cv2.imread(ori_img_path,  cv2.IMREAD_GRAYSCALE)
        result = ''
        for idx, line in enumerate(fh1):
            data = (line.replace("\n", "")).split(" ")
            if float(data[1]) < conf_thres_detector: #ignore boudingbox wit`h low confident
                continue
            xmin = int(data[2])
            ymin = int(data[3])
            width = int(data[4])
            height = int(data[5])
            coords = [xmin, ymin, xmin + width, ymin + height]
            roi = crop_from_img_by_margin(img, coords, 0.05)
            char_data = preprocess_img(roi)
            result_classifier = model_classifier.predict(char_data)
            max_idx_classifier = np.argmax(result_classifier, axis=1)
            max_val_classifier = np.amax(result_classifier)
            if max_val_classifier < conf_thres_classifier: #ignore roi that has low confident
                continue

            final_pred_char = class_id_vs_name[max_idx_classifier[0]]
#   chungnx add ignore background class
            if final_pred_char != 'background':
                result_line = final_pred_char + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + data[5]
                result += result_line + '\n'
                print(result_line)

        with open(os.path.join(output_dir, file_name+'.txt'), "w") as f:
            f.write(result)

    print("Done.")


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(output_dir)
    pred_from_2nd_stage()
