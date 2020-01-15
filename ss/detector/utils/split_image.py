import os
import cv2
import datetime
import time
import math
from glob import glob

import numpy as np


def split_image(imgage_obj, img_shape=(300, 300), overap_size=100,
                crop_path='./cropped/'):
    FILE_SEPERATOR = os.path.sep

    '''
    split_size_height = int(img_shape[1] / zoom_ratio)
    split_size_width = int(img_shape[0] / zoom_ratio)
    '''
    split_size_height = img_shape[1]
    split_size_width = img_shape[0]

    if not os.path.isdir(crop_path):
        os.mkdir(crop_path)

    file_name = 'img_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # file_name = "".join(image_path.split(FILE_SEPERATOR)[-1].split('.')[:-1])
    # print(file_name)

    save_dir_name = crop_path + FILE_SEPERATOR + str(file_name + FILE_SEPERATOR)
    if not os.path.isdir(save_dir_name):
        os.mkdir(save_dir_name)

    if imgage_obj is None:
        raise FileNotFoundError()

    height, width = imgage_obj.shape[:2]

    idx_w = math.ceil(width / (split_size_width - overap_size))
    idx_h = math.ceil(height / (split_size_height - overap_size))
    count = 0
    for j in range(0, idx_h):
        for i in range(0, idx_w):
            start_x = i * (split_size_width - overap_size)
            start_y = j * (split_size_height - overap_size)

            if width - start_x > overap_size and height - start_y > overap_size:
                start_x = start_x if start_x < (width - split_size_width) else width - split_size_width
                start_y = start_y if start_y < (height - split_size_height) else height - split_size_height
                end_x = start_x + split_size_width if (start_x + split_size_width) < width else width
                end_y = start_y + split_size_height if (start_y + split_size_height) < height else height

                roi = imgage_obj[start_y:end_y, start_x:end_x]
                count += 1
                tmp = save_dir_name + file_name + "_" + str(count).zfill(4) + "_" + str(width) + "_" + str(height) \
                      + "_" + str(split_size_width) + "_" + str(split_size_height) \
                      + "_" + str(start_x) + "_" + str(start_y) + "_" + str(end_x) + "_" + str(end_y) + ".png"
                '''
                # resize image 
                if zoom_ratio != 1.0 :                            
                    roi = cv2.resize(roi, (img_shape[0], img_shape[1]), 
                                     interpolation=cv2.INTER_CUBIC)
                '''
                '''
                math_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                math_close = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, math_kernel)
                math_div = roi / math_close
                math_res = np.zeros(math_div.shape)
                cv2.normalize(math_div, math_res, 0, 255, cv2.NORM_MINMAX)                


                #roi = contrastStretching(math_res)
                '''
                # save image
                cv2.imwrite(tmp, roi)

    return save_dir_name


def split_image_to_objs(imgage_obj, img_shape=(300, 300), overap_size=100, zoom_ratio=None):
    split_size_height = img_shape[1]
    split_size_width = img_shape[0]

    img_obj_list = []
    img_coord_list = []

    if imgage_obj is None:
        raise FileNotFoundError()

    height, width = imgage_obj.shape[:2]

    idx_w = math.ceil(width / (split_size_width - overap_size))
    idx_h = math.ceil(height / (split_size_height - overap_size))
    count = 0
    for j in range(0, idx_h):
        for i in range(0, idx_w):
            start_x = i * (split_size_width - overap_size)
            start_y = j * (split_size_height - overap_size)

            if width - start_x > overap_size and height - start_y > overap_size:
                start_x = start_x if start_x < (width - split_size_width) else width - split_size_width
                start_y = start_y if start_y < (height - split_size_height) else height - split_size_height
                end_x = start_x + split_size_width if (start_x + split_size_width) < width else width
                end_y = start_y + split_size_height if (start_y + split_size_height) < height else height

                img_obj_list.append(imgage_obj[start_y:end_y, start_x:end_x])
                img_coord_list.append([start_x, start_y, end_x, end_y, zoom_ratio])

    return img_obj_list, img_coord_list



