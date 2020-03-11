import math
import os
from datetime import datetime
from os import listdir

import cv2
import numpy as np

from classifier_CRNN.pre_processing.table_border_extraction_fns import get_h_and_v_line_bbox_CNX, detect_table
from .image_calibration import calib_image

kernel_sharpening = np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [-1, -1, -1, -1, 17, -1, -1, -1, -1],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 0, -1, 0, 0, 0, 0]])

kernel_blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9]])

kernel_shapping2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, scaleWunderH=np.sqrt(2)):
    # obtain a consistent order of the points and unpack them
    # individually
    pts = np.array(pts, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_AREA)
    # resize A4
    warped = cv2.resize(warped, (maxWidth, int(maxWidth * scaleWunderH)), interpolation=cv2.INTER_NEAREST)

    # return the warped image
    return warped


def rotate_image_angle(img, angle, pixel_erase=1):
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # rotate image
    abs_cos = abs(Mat_rotation[0, 0])
    abs_sin = abs(Mat_rotation[0, 1])

    bound_w = int(h_org * abs_sin + w_org * abs_cos)
    bound_h = int(h_org * abs_cos + w_org * abs_sin)

    Mat_rotation[0, 2] += bound_w / 2 - w_org / 2
    Mat_rotation[1, 2] += bound_h / 2 - h_org / 2

    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))
    blank_image = np.zeros(shape=[h_org, w_org, 1], dtype=np.uint8)
    blank_image = 255 - blank_image
    img_result_blank = cv2.warpAffine(blank_image, Mat_rotation, (bound_w, bound_h))
    img_result_blank = 255 - img_result_blank
    img_result = erase_img(img_result, img_result_blank, None, 1)
    return img_result


def flip_image(src, flip):
    # flip 0 flip up down, 1 flip left right, -1 flip up down left right
    result = None
    result = cv2.flip(src, flip)
    return result


def invert_image(src):
    h, w, c = src.shape
    gray = None
    if (c == 3):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # convert to binary
    adaptivethreshold_applied_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 41, 7)
    invert_img = 255 - adaptivethreshold_applied_img
    return invert_img


def erase_img(img, temp, rect, pixel_cal=5, method=cv2.INPAINT_TELEA):
    # format  of rect [x1, y1, x2, y2]
    # (x1, y1) coordinate top left, (x2, y2) coordinate bottom right
    dst = None
    if temp is None:
        h = img.shape[0]
        w = img.shape[1]
        x1, y1, x2, y2 = rect
        blank_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
        for r in range(y1, (y2 + 1)):
            for c in range(x1, (x2 + 1)):
                if r < h and c < w:
                    blank_image[r, c] = 255
        # mask = cv2.threshold(blank_image,100,255,cv2.THRESH_BINARY)
        dst = cv2.inpaint(img, blank_image, pixel_cal, method)
    else:
        dst = cv2.inpaint(img, temp, pixel_cal, method)
    return dst


def auto_rotation(img, expand_angle=5):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray_img.shape
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    blank_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        list_bb.append([x, y, x + w, y + h])
    img_cp = img.copy()
    img_cp2 = img.copy()
    for b in list_bb:
        centerx = int((b[2] - b[0]) / 2 + b[0])
        centery = int((b[3] - b[1]) / 2 + b[1])
        cv2.circle(blank_image, (centerx, centery), 2, (255, 255, 0), -1)
    lines = cv2.HoughLinesP(blank_image, 3, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=35)
    list_angle = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = (math.atan2((y1 - y2), (x1 - x2)) * 180) / math.pi
        list_angle.append(angle)
    list_cluster_angle = []
    max_candidate = 0
    angle_rotate = 0
    for ag in list_angle:
        hascluster = False
        for cluster in list_cluster_angle:
            if abs(ag - cluster[0]) < expand_angle:
                cluster[0] = (cluster[0] * cluster[1] + ag) / (cluster[1] + 1)
                cluster[1] += 1
                if cluster[1] > max_candidate:
                    max_candidate = cluster[1]
                    angle_rotate = cluster[0]
                hascluster = True
                break
        if hascluster == False:
            new_cluster = [ag, 1]
            list_cluster_angle.append(new_cluster)
    convert_angle_opencv = 180 - angle_rotate
    img_cp = rotate_image_angle(img_cp, convert_angle_opencv)
    return img_cp


def crop_image(img, x1, y1, x2, y2, offsetx=0, offsety=0):
    return img[int(y1 + offsety):int(y2 + offsety), int(x1 + offsetx):int(x2 + offsetx)]


def get_coor_match_template(img, template, type_img=0, numb_scale=50):
    # type 0 rgb, 1 gray, 3 binary
    img_process = img.copy()
    temp_process = template.copy()
    if type_img == 1:
        img_process = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_process = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    elif type_img == 3:
        img_process = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_process = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_process = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 41, 7)
        temp_process = cv2.adaptiveThreshold(temp_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 41, 7)
    h_temp = template.shape[0]
    w_temp = template.shape[1]
    low_scale = max(template.shape[1] / img.shape[1], template.shape[0] / img.shape[0])
    lscales = np.linspace(1, low_scale, num=numb_scale)
    found = None
    for scale in lscales:
        imrs = cv2.resize(img_process, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h_imrs = imrs.shape[0]
        w_imrs = imrs.shape[1]
        if h_imrs < h_temp or w_imrs < w_temp:
            break
        res = cv2.matchTemplate(imrs, temp_process, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if found is None or max_val > found[0]:
            found = [max_val, max_loc[0], max_loc[1], scale]
    print("max val : ", max_val)
    top_left = [found[1], found[2]]
    top_left[0] /= found[3]
    top_left[1] /= found[3]
    w = w_temp / found[3]
    h = h_temp / found[3]
    coor_temp1 = [int(top_left[0]), int(top_left[1]), int(top_left[0] + w), int(top_left[1] + h)]
    return coor_temp1


class clTemplate:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.name_template = ''
        self.listBoxtemplate = []
        self.listBoxinfor = []


def auto_extract_info(img, path_config_file, threshold=0.75):
    list_template = []
    list_filebackground = os.listdir(path_config_file)
    print(list_filebackground)
    if len(list_filebackground) == 0:
        print("template files empty")
        return
    else:
        for i in list_filebackground:
            template = clTemplate()
            path_f = os.path.join(path_config_file, i)
            with open(path_f, 'r+') as readf:
                count = 0
                for line in readf:
                    count += 1
                    if count == 1:
                        template.name_template = line
                    else:
                        list_inf = line.split()
                        if len(list_inf) == 7:
                            bb_t = [int(list_inf[0]), int(list_inf[1]), int(list_inf[2]), \
                                    int(list_inf[3]), float(list_inf[4]), float(list_inf[5]), float(list_inf[6])]
                            template.listBoxtemplate.append(bb_t)
                        elif len(list_inf) == 4:
                            bb_i = [int(list_inf[0]), int(list_inf[1]), int(list_inf[2]), int(list_inf[3])]
                            template.listBoxinfor.append(bb_i)
                        elif len(list_inf) == 2:
                            template.height = int(list_inf[0])
                            template.width = int(list_inf[1])
            list_template.append(template)
    img_bl = auto_rotation(img)
    im_bl_gray = cv2.cvtColor(img_bl, cv2.COLOR_BGR2GRAY)
    h_n, w_n = im_bl_gray.shape
    im_edges = cv2.Canny(im_bl_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    center_img = [int(w_n / 2), int(h_n / 2)]
    if major == '3':
        _, contours, _ = cv2.findContours(im_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(im_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb_img = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 15 and h > 15):
            center_bb = [int(x + w / 2), int(y + h / 2)]
            # ratio_center = (center_tmp[0] - center_bb[0]) / (center_tmp[1] - center_bb[1])
            ratio_w = (center_img[0] - center_bb[0]) / w
            ratio_h = (center_img[1] - center_bb[1]) / h
            list_bb_img.append([x, y, w, h, w / h, ratio_w, ratio_h])
    list_matching_temp = []
    for templ in list_template:
        list_bb_tmp = templ.listBoxtemplate
        if (len(list_bb_img) / len(list_bb_tmp)) < threshold:
            list_matching_temp.append(0)
            continue
        outlier_boxes = []
        for tmp_bb in list_bb_tmp:
            outlier = True
            for img_bb in list_bb_img:
                if tmp_bb[4] != 0 and tmp_bb[5] != 0 and tmp_bb[6] != 0:
                    if (img_bb[4] / tmp_bb[4]) > 0.8 and (img_bb[4] / tmp_bb[4]) < 1.2 \
                            and (img_bb[5] / tmp_bb[5]) > 0.8 and (img_bb[5] / tmp_bb[5]) < 1.2 \
                            and (img_bb[6] / tmp_bb[6]) > 0.8 and (img_bb[6] / tmp_bb[6]) < 1.2:
                        outlier = False
                        break
            if outlier == True:
                outlier_boxes.append(tmp_bb)
        list_matching_temp.append(len(outlier_boxes) / len(list_bb_tmp))
    confi_match = min(list_matching_temp)
    print("conf: ", confi_match)
    if (1 - confi_match) < threshold:
        print("not match")
        return
    id_temp_match = list_matching_temp.index(confi_match)
    template_m = list_template[id_temp_match]
    ratioy = h_n / template_m.height
    ratiox = w_n / template_m.width
    count = 0
    for if_box in template_m.listBoxinfor:
        count += 1
        bx = int(if_box[0] * ratiox)
        by = int(if_box[1] * ratioy)
        ex = int(if_box[2] * ratiox)
        ey = int(if_box[3] * ratioy)
        info_img = crop_image(img_bl, bx, by, ex, ey)
        cv2.imwrite("crop_info" + str(count) + ".jpg", info_img)
        cv2.waitKey()


class clTemplate_demo:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.name_template = ''
        self.listBoxinfor = []


class clImageInfor_demo:
    def __init__(self):
        self.data = None
        self.prefix = ""
        self.type = ""
        self.nameTemplate = ""
        self.location = []
        self.value = ""
        self.value_nlp = ""


def background_subtract(image, bgr_path=''):
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # background=cv2.imread(bgr_path, 0)
    background = cv2.imread(bgr_path)
    try:
        result = cv2.subtract(background, image)
        result_inv = cv2.bitwise_not(result)
        # cv2.imshow('result',result_inv)
        # cv2.waitKey(0)
        return result_inv
    except:
        return image


def gen_image_for_demo(path_image,
              save_path='/data/data_imageVIB/1/',
              save_filename='result.jpg',
              path_config_file='classifier_CRNN/form/template_VIB_page1_demo.txt',
              eraseline=False, subtract_bgr=True):
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    template = clTemplate_demo()
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            if count == 1:
                template.name_template = line
            else:
                list_inf = line.split()
                if len(list_inf) == 6:
                    bb_i = [list_inf[0],list_inf[1],int(list_inf[2]), int(list_inf[3]), int(list_inf[2]) + int(list_inf[4]),int(list_inf[3]) + int(list_inf[5])]
                    template.listBoxinfor.append(bb_i)
                elif len(list_inf) == 2:
                    template.height = int(list_inf[0])
                    template.width = int(list_inf[1])
    count_img = 0
    list_class_img_info = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_bl = cv2.imread(path_image)
    img_bl = cv2.resize(img_bl, (2480, 3508))
    img_bl = calib_image(img_bl)

    img_save_bl = img_bl.copy()
    h_n = img_bl.shape[0]
    w_n = img_bl.shape[1]
    ratioy = h_n / template.height
    ratiox = w_n / template.width
    for if_box in template.listBoxinfor:
        classimginf = clImageInfor_demo()
        classimginf.prefix = if_box[0]
        classimginf.type = if_box[1]
        classimginf.nameTemplate = template.name_template
        count_img += 1
        bx = int(if_box[2] * ratiox)
        by = int(if_box[3] * ratioy)
        ex = int(if_box[4] * ratiox)
        ey = int(if_box[5] * ratioy)
        cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
        cv2.putText(img_save_bl, if_box[1], (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        classimginf.location = [bx,by,ex,ey]
        info_img = crop_image(img_bl, bx, by, ex, ey, offsety=1, offsetx=1)
        if subtract_bgr:
            info_img = background_subtract(info_img, bgr_path='form/'+if_box[0]+'.jpg')
        if eraseline == True:
            rs = removeLines(info_img,expand_bb=3,pixel_erase=2)
            if rs is not None:
                info_img = rs
        classimginf.data = info_img
        save_path_rs = os.path.join(save_path, save_filename)
        list_class_img_info.append(classimginf)
        cv2.imwrite(save_path_rs, img_save_bl)

def extract_for_demo(listimg, path_config_file, save_to_database=False, eraseline=False, subtract_bgr=True, gen_bgr=False):
    if len(listimg) == 0:
        print("input image empty")
        return
    template = clTemplate_demo()
    max_wh_ratio = 1
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            if count == 1:
                template.name_template = line
            else:
                list_inf = line.split()
                if len(list_inf) == 6:
                    bb_i = [list_inf[0], list_inf[1], int(list_inf[2]), int(list_inf[3]),
                            int(list_inf[2]) + int(list_inf[4]), int(list_inf[3]) + int(list_inf[5])]
                    wh_ratio = float(list_inf[4]) / float(list_inf[5])
                    #print('wh_ratio',wh_ratio)
                    if wh_ratio > max_wh_ratio:
                        max_wh_ratio = wh_ratio
                    template.listBoxinfor.append(bb_i)
                elif len(list_inf) == 2:
                    template.height = int(list_inf[0])
                    template.width = int(list_inf[1])
    #print('max wh ratio',max_wh_ratio)
    count_img = 0
    list_class_img_info = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img in listimg:
        img_bl = img
        img_save_bl = img_bl.copy()
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        ratioy = h_n / template.height
        ratiox = w_n / template.width
        for if_box in template.listBoxinfor:
            classimginf = clImageInfor_demo()
            classimginf.prefix = if_box[0]
            classimginf.type = if_box[1]
            classimginf.nameTemplate = template.name_template
            count_img += 1
            bx = int(if_box[2] * ratiox)
            by = int(if_box[3] * ratioy)
            ex = int(if_box[4] * ratiox)
            ey = int(if_box[5] * ratioy)
            if save_to_database:
                cv2.rectangle(img_save_bl, (bx, by), (ex, ey), (0, 0, 255), 4)
                cv2.putText(img_save_bl, if_box[1], (bx - 100, by + 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            classimginf.location = [bx, by, ex, ey]
            offsetx, offsety = 1, 1
            if gen_bgr:
                offsetx, offsety = 0, 0
            info_img = crop_image(img_bl, bx, by, ex, ey, offsety=offsety, offsetx=offsetx)
            if gen_bgr:
                cv2.imwrite('form/background_VIB/' + if_box[0] + '.jpg', info_img)
            if subtract_bgr:
                info_img = background_subtract(info_img, bgr_path='form/background_VIB/' + if_box[0] + '.jpg')
            if eraseline == True:
                rs = removeLines(info_img, expand_bb=3, pixel_erase=2)
                if rs is not None:
                    info_img = rs
            classimginf.data = info_img
            list_class_img_info.append(classimginf)
        if save_to_database:
            cv2.imwrite("/data/data_imageVIB/result.jpg", img_save_bl)
    return list_class_img_info, max_wh_ratio


def check_mark(img, scale_box_check=0.7):
    print("--------------------------------------------------------------------------------------")
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img_iv = invert_image(img)
    cv2.imshow("check", img_iv)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, he = cv2.findContours(img_iv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else:
        contours, he = cv2.findContours(img_iv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.rectangle(img,(x,y),(x + w, y + h),(0,0,255),1)
    # cv2.imshow("mark",img)
    list_contour_rec = []
    smax = 0
    top_l = None
    btm_r = None
    size_max = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 5 or h > 5):
            if w * h > smax:
                smax = w * h
                top_l = [x, y]
                btm_r = [x + w, y + h]
                size_max = [w, h]
    if size_max[0] / size_max[1] > 1.1 or size_max[0] / size_max[1] < 0.8:
        print("checked mark")
        return True
    else:
        w_check = size_max[0] * scale_box_check
        h_check = size_max[1] * scale_box_check
        print()
        x_center = top_l[0] + size_max[0] / 2
        y_center = top_l[1] + size_max[1] / 2
        top_l_check = [int(x_center - w_check / 2), int(y_center - h_check / 2)]
        print(top_l_check)
        btm_r_check = [int(top_l_check[0] + w_check), int(top_l_check[1] + h_check)]
        print(btm_r_check)
        imcrop_b = crop_image(img_iv, top_l_check[0], top_l_check[1], btm_r_check[0], btm_r_check[1])
        numb_w = cv2.countNonZero(imcrop_b)
        cv2.rectangle(img, (top_l[0], top_l[1]), (btm_r[0], btm_r[1]), (0, 0, 255), 2)
        cv2.rectangle(img, (top_l_check[0], top_l_check[1]), (btm_r_check[0], btm_r_check[1]), (255, 0, 0), 2)
        cv2.imshow("img_", img)
        cv2.imshow("img_crop_b", imcrop_b)
        cv2.waitKey()
        print("W: ", numb_w)
        if numb_w / (w_check * h_check) >= 0.05:
            print("checked mark")
            return True
        else:
            print("unchecked mark1")
            return False


def removeLines(img, margin=5, expand_bb=2, b_erase_img=True, pixel_erase=3):
    img_cp = img.copy()
    edges = invert_image(img)
    img_cp2 = img.copy()
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 3, np.pi / 180, threshold=90, minLineLength=60, maxLineGap=10)
    list_cluster_lines = []
    max_lenght = 0
    max_width = 0
    bb_line = []
    list_line_max = []
    # struct cluster y average, numb point incluster, x top right ,y top right, x bottom left, y bottom left
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        cv2.line(img_cp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ytb = int((y1 + y2) / 2)
        hascluster = False
        for cluster in list_cluster_lines:
            if abs(ytb - cluster[0]) < margin:
                cluster[0] = (cluster[0] * cluster[1] + ytb) / (cluster[1] + 1)
                cluster[1] += 1
                cluster[2] = min(cluster[2], x1, x2)
                cluster[3] = min(cluster[3], y1, y2)
                cluster[4] = max(cluster[4], x1, x2)
                cluster[5] = max(cluster[5], y1, y2)
                cluster[6].append([x1, y1, x2, y2])
                if (cluster[4] - cluster[2]) > max_lenght and (cluster[5] - cluster[3]) >= max_width:
                    max_lenght = (cluster[4] - cluster[2])
                    max_width = (cluster[5] - cluster[3])
                    bb_line = [cluster[2], cluster[3] - expand_bb, cluster[4], cluster[5] + expand_bb]
                    list_line_max = cluster[6]
                hascluster = True
                break
        if hascluster == False:
            new_cluster = [ytb, 1, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            list_line = []
            list_line.append([x1, y1, x2, y2])
            new_cluster.append(list_line)
            list_cluster_lines.append(new_cluster)
            if (new_cluster[4] - new_cluster[2]) >= max_lenght and (new_cluster[5] - new_cluster[3]) >= max_width:
                max_lenght = (new_cluster[4] - new_cluster[2])
                max_width = (new_cluster[5] - new_cluster[3])
                list_line_max = new_cluster[6]
                bb_line = [new_cluster[2], new_cluster[3] - expand_bb, new_cluster[4], new_cluster[5] + expand_bb]
    if (bb_line[2] - bb_line[0]) / img.shape[2] > 0.7:
        if b_erase_img == True:
            blank_image = np.zeros(shape=[img_cp2.shape[0], img_cp2.shape[1], 1], dtype=np.uint8)
            img_test = img.copy()
            for ln in list_line_max:
                cv2.line(img_test, (ln[0], ln[1]), (ln[2], ln[3]), (0, 0, 255), expand_bb)
                cv2.line(blank_image, (ln[0], ln[1]), (ln[2], ln[3]), (255, 255, 255), expand_bb)
            img_cp2 = erase_img(img_cp2, blank_image, bb_line, pixel_erase, cv2.INPAINT_NS)
        else:
            crop_img = crop_image(img_cp2, 0, 0, bb_line[2] - bb_line[0], bb_line[3] - bb_line[1])
            img_cp2[bb_line[1]:bb_line[3], bb_line[0]:bb_line[2]] = crop_img
        return img_cp2
    else:
        return None


def crop_fit_text(img, expand=5):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    image_i = invert_image(img)
    lines = cv2.HoughLinesP(image_i, 3, np.pi / 180, threshold=90, minLineLength=10, maxLineGap=3)
    blank_image = np.zeros(shape=[image_i.shape[0], image_i.shape[1], 1], dtype=np.uint8)
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
    h_img_b = cv2.morphologyEx(blank_image, cv2.MORPH_ERODE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 3))
    h_img_b = cv2.morphologyEx(h_img_b, cv2.MORPH_DILATE, kernel,
                               iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))
    v_img = cv2.morphologyEx(blank_image, cv2.MORPH_ERODE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 70))
    v_img = cv2.morphologyEx(v_img, cv2.MORPH_DILATE, kernel,
                             iterations=3)
    blank_image = h_img_b + v_img
    img_erase = erase_img(img, blank_image, 5)

    img_gray = cv2.cvtColor(img_erase, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    xtop_left = 10e9
    ytop_left = 10e9
    xbt_right = -1
    ybt_right = -1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if w / h < 3 and h / w < 3 and w > 10 and h > 10:
            if x < xtop_left:
                xtop_left = x
            if y < ytop_left:
                ytop_left = y
            if x + w > xbt_right:
                xbt_right = x + w
            if y + h > ybt_right:
                ybt_right = y + h
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
    img_crop = crop_image(img_erase, max(0, xtop_left - expand), max(0, ytop_left - expand), \
                          min(xbt_right + expand, w_img), min(ybt_right + expand, h_img))
    return img_crop


def list_files1(directory, extension):
    # print(listdir(directory))
    list_file = []
    for f in listdir(directory):
        if f.endswith('.' + extension):
            list_file.append(f)
    return list_file


def store_data_handwriting_template(path, path_config_file, save_path, expand_y=0):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.dirname(path)

    pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    if save_path is not None:
        result_save_dir = os.path.join(save_path, "aicrhw_" + pred_time)
        if not os.path.isdir(result_save_dir):
            os.mkdir(result_save_dir)
    if len(list_img) == 0:
        print("input image empty")
        return
    template = clTemplate_demo()
    with open(path_config_file, 'r+') as readf:
        count = 0
        for line in readf:
            count += 1
            if count == 1:
                template.name_template = line
            else:
                list_inf = line.split()
                if len(list_inf) == 5:
                    bb_i = [list_inf[0], int(list_inf[1]), int(list_inf[2]), int(list_inf[1]) + int(list_inf[3]),
                            int(list_inf[2]) + int(list_inf[4])]
                    template.listBoxinfor.append(bb_i)
                elif len(list_inf) == 2:
                    template.height = int(list_inf[0])
                    template.width = int(list_inf[1])
    count_img = 0
    list_class_img_info = []
    count = 0
    for path_img in list_img:
        path_img = os.path.join(path, path_img)
        img = cv2.imread(path_img)
        count += 1
        path_save_image = os.path.join(result_save_dir, "AICR_P0000" + str(count))
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)
        path_org_img = os.path.join(path_save_image, "origine.jpg")
        cv2.imwrite(path_org_img, img)
        img_bl = auto_rotation(img)
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        ratioy = h_n / template.height
        ratiox = w_n / template.width
        for if_box in template.listBoxinfor:
            prefix = if_box[0]
            count_img += 1
            bx = int(if_box[1] * ratiox)
            by = int((if_box[2] - expand_y) * ratioy)
            ex = int(if_box[3] * ratiox)
            ey = int((if_box[4] + expand_y) * ratioy)
            info_img = crop_image(img_bl, bx, by, ex, ey)
            save_path_img = os.path.join(path_save_image, prefix + ".jpg")
            cv2.imwrite(save_path_img, info_img)
    print("finished !!")


def store_data_handwriting_table(path, save_path, expand_y=0):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.dirname(path)

    pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
    if save_path is not None:
        result_save_dir = os.path.join(save_path, "aicrhw_" + pred_time)
        if not os.path.isdir(result_save_dir):
            os.mkdir(result_save_dir)
    if len(list_img) == 0:
        print("input image empty")
        return
    list_class_img_info = []
    count = 0
    for path_img in list_img:
        path_img = os.path.join(path, path_img)
        img = cv2.imread(path_img)
        count += 1
        path_save_image = os.path.join(result_save_dir, "AICR_P0000" + str(count))
        if not os.path.isdir(path_save_image):
            os.mkdir(path_save_image)
        path_org_img = os.path.join(path_save_image, "origine.jpg")
        cv2.imwrite(path_org_img, img)
        img_bl = img.copy()
        h_n = img_bl.shape[0]
        w_n = img_bl.shape[1]
        hline_list, vline_list = get_h_and_v_line_bbox_CNX(img_bl)
        list_p_table, hline_list, vline_list = detect_table(hline_list, vline_list)
        count_img = 0
        for table in list_p_table:
            table.detect_cells()
            len_cells = len(table.listCells)
            if len_cells != 12:
                print("error ", path_save_image)
            for id_box in range(len_cells):
                if id_box % 2 != 0:
                    count_img += 1
                    bx, by, ex, ey = table.listCells[id_box]
                    info_img = crop_image(img_bl, bx, by, ex, ey)
                    save_path_img = os.path.join(path_save_image, str(count_img) + ".jpg")
                    cv2.imwrite(save_path_img, info_img)
    print("finished !!")


if __name__ == "__main__":
    # list_img = []
    # img = cv2.imread("C:/Users/chungnx/Desktop/img_user/0001.jpg")
    # list_img.append(img)
    # img = cv2.imread("2020-02-25_16-00/NAME1.jpg")
    # list_img.append(img)
    # img2 = augment_random_erase(img,5)
    # cv2.imshow("img2 ",img2)
    # cv2.waitKey()
    # img2 = four_point_transform(img, [[94, 79], [953, 65], [959, 1258], [102, 1257]])
    # cv2.imwrite("im2.jpg", img2)
    # augumentation_image(img,3)
    # gen_data_path("/data/dataset/cinnamon_data/0916_DataSamples/","/data/dataset/cinnamon_data")
    # gen_data_path("/data/dataset/cinnamon_data/1015_Private Test/", "/data/dataset/cinnamon_data")
    # extract_for_demo(list_img,"template_VIB_page1.txt","",True)
    # path = "C:/Users/chungnx/Desktop/checked_mark.jpg"
    # img = cv2.imread(path)
    # check_mark(img)
    path = "C:/Users/chungnx/Desktop/check_mark"
    path_bg = "C:/Users/chungnx/Desktop/background_img/shk.jpg"
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    img_bg = cv2.imread(path_bg)
    for l in list_img:
        path_img = os.path.join(path, l)
        img = cv2.imread(path_img)
        check_mark(img)
        cv2.waitKey()
    # path = "C:/Users/chungnx/Desktop/IMG_20200303_091222.jpg"
    # img = cv2.imread(path)
    # img_bl = auto_rotation(img)
    # feature1 = "C:/Users/chungnx/Desktop/temp_standart/feature1.jpg"
    # feature2 = "C:/Users/chungnx/Desktop/temp_standart/feature2.jpg"
    # temp = cv2.imread(feature1)
    # coor_1 = get_coor_match_template(img_bl,temp,type_img = 1)
    # temp = cv2.imread(feature2)
    # coor_2 = get_coor_match_template(img_bl,temp,type_img = 1)
    # img_bl2 = img_bl.copy()
    # # coor_3 = get_coor_match_template(img, temp)
    # cv2.rectangle(img_bl2, (coor_1[0], coor_1[1]), (coor_1[2], coor_1[3]),(0, 0, 255),3)
    # cv2.rectangle(img_bl2, (coor_2[0], coor_2[1]), (coor_2[2], coor_2[3]), (0, 0, 255), 3)
    #     # cv2.imwrite("img_bl2.jpg",img_bl2)
    #     # fp = [[coor_1[0], coor_1[1]], [coor_1[2], coor_1[1]], [coor_2[0], coor_2[3]], [coor_2[2], coor_2[3]]]
    #     # img_4 = img_per = four_point_transform(img_bl,fp)
    #     # cv2.imwrite("img_4.jpg",img_4)
    #     # top_left = [coor_1[0],coor_1[1]]
    #     # top_right = [coor_1[2],coor_1[1]]
    #     # bt_left = [coor_3[0],coor_3[3]]
    #     # cv2.imwrite("imgrs.jpg",img)
    #     # cv2.waitKey()
    #     # path_configfile = "venv/template_data.txt"
    #     # path_save = "C:/Users/chungnx\Desktop/aicr_data_hw/image_crop"
    #     # store_data_handwriting_table(path_img,path_save,0)
    #     # crop_fit_text(img)
