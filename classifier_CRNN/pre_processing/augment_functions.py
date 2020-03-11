import cv2
import numpy as np
import torch
from PIL import Image
import pickle
import math
import os
from os import listdir
from datetime import datetime
import random
import shutil
from torchvision import transforms
try:
    from pre_processing.image_preprocessing import erase_img, crop_image, rotate_image_angle, invert_image, list_files1
    from pre_processing.table_border_extraction_fns import get_h_and_v_line_bbox, \
    clTable, get_h_and_v_line_bbox_CNX, filter_lines_error, detect_table
except ImportError:
    from image_preprocessing import erase_img, crop_image, rotate_image_angle, invert_image, list_files1
    from table_border_extraction_fns import get_h_and_v_line_bbox, \
        clTable, get_h_and_v_line_bbox_CNX, filter_lines_error, detect_table


def augment_image_addline(img, color = (60, 60, 60),type = 0,size_draw  = 3, range_dot = 15, sizelinedot = 1):
    #type 0 solid line, 1 dots line,2 line dot
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sum_h = 0
    count = 0
    listbb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if(w > 20 and h > 20):
            sum_h += (y+h)
            count+=1
            listbb.append([x,y,x+w,y+h])
    h_average = int(sum_h/count)
    if type == 0:
        cv2.line(img2, (0, h_average-1), (w_img, h_average-1), color, int(size_draw/3), cv2.LINE_AA)
        cv2.line(img2, (0, h_average), (w_img, h_average), color, int(size_draw/3), cv2.LINE_AA)
        cv2.line(img2, (0, h_average+1), (w_img, h_average+1), color, int(size_draw/3),  cv2.LINE_AA )
    elif type == 1 or type == 2:
        for i in range(int(w_img/range_dot)):
            if type == 1:
                centerx = int((i*range_dot) + int(size_draw/2))
                centery = h_average - int((size_draw)/2)
                cv2.circle(img2,(centerx,centery),size_draw,color,-1)
            elif type == 2:
                size_line = int(range_dot/sizelinedot)
                beginl = int(i*range_dot) + int(i* size_line)
                endl   = beginl+size_line
                cv2.line(img2, (beginl, h_average - 1), (endl, h_average - 1), color, int(size_draw / 3), cv2.LINE_AA)
                cv2.line(img2, (beginl, h_average), (endl, h_average), color, int(size_draw / 3), cv2.LINE_AA)
                cv2.line(img2, (beginl, h_average + 1), (endl, h_average + 1), color, int(size_draw / 3), cv2.LINE_AA)
    return img2

def augment_random_rotate(img,begin, end, pixel_erase = 3):
    angle = random.randrange(begin, end)
    img2 = rotate_image_angle(img, angle)
    return img2

def augment_random_erase(img, numb_char_erase):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sum_h = 0
    count = 0
    listbb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 20 and h > 20):
            sum_h += (y + h)
            count += 1
            listbb.append([x, y, x + w, y + h])
    id_box_erase = random.sample(range(0, len(listbb)-1), numb_char_erase)
    print(id_box_erase)
    for id in id_box_erase:
        xb, yb, xe, ye = listbb[id]
        xb = random.randrange(xb, int((xe + xb) / 2))
        xe = random.randrange(xb + 1, xe)
        yb = random.randrange(yb, ye)
        img2 = erase_img(img2, None, [xb, yb, xe, ye], 10)
    return img2

def augment_resize_img(img, w, h, fsx = None, fsy = None):
    if fsx is None and fsy is None:
        imgrs = cv2.resize(img,(w,h),interpolation = cv2.INTER_AREA)
        return imgrs
    else:
        imgrs = cv2.resize(img, None , fx = fsx, fy = fsy, interpolation=cv2.INTER_AREA)
        return imgrs

def augment_bold_characters(img, kernel_dms = 3, iter = 1):
    kernel = np.ones((kernel_dms, kernel_dms), np.uint8)
    img_bold = cv2.erode(img, kernel, iterations=iter)
    return img_bold

def augment_thin_characters(img, kernel_dms = 3, iter = 1):
    kernel = np.ones((kernel_dms, kernel_dms), np.uint8)
    img_thin = cv2.dilate(img, kernel, iterations=iter)
    return img_thin

def augment_blur(img, type_blur = 0, size_window = 5):
    #type 0 average bluer, 1 gaussian blur, 2 median blur
    img_rs = None
    if type_blur == 0:
        img_rs = cv2.blur(img,(size_window,size_window))
    elif type_blur == 1:
        img_rs = cv2.GaussianBlur(img,(size_window,size_window),0)
    else:
        img_rs = cv2.medianBlur(img, size_window)
    return img_rs

def augment_add_noise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

def aument_gen_list_box_number(img,img_bg = None, expand_size = 10, expand_size_char = 3, size_line_board = 2,color_text = None,):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    img_edges = invert_image(img)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb_ct = []

    img_draw = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
            list_bb_ct.append([x,y,x+w,y+h, w, h])
    #clean bb
    len_list = len(list_bb_ct)
    check_list = [0] * len_list
    for i in range(len_list):
        if check_list[i] == 1:
            continue
        bx1, by1, ex1, ey1, _, _ = list_bb_ct[i]
        for j in range(i+1,len_list):
            if check_list[j] == 1:
                continue
            bx2, by2, ex2, ey2,_,_ = list_bb_ct[j]
            if bx1 >= bx2 and by1 >= by2 \
               and ex1 <= ex2 and ey1 <= ey2:
                    check_list[i] = 1
                    break
            elif bx2 >= bx1 and by2 >= by1 \
               and ex2 <= ex1 and ey2 <= ey1:
                    check_list[j] = 1

    list_bb_final = []
    wmax = 0
    hmax = 0
    for i in range(len_list):
        if check_list[i] == 0:
            list_bb_final.append(list_bb_ct[i])
            if list_bb_ct[i][4] > wmax:
                wmax = list_bb_ct[i][4]
            if list_bb_ct[i][5] > hmax:
                hmax = list_bb_ct[i][5]
    list_bb_final.sort(key=lambda x:x[0])
    len_final = len(list_bb_final)
    max_size = max(hmax,wmax) + expand_size_char
    w_img = len_final*max_size + (len_final+1)*size_line_board + expand_size*2
    h_img = max_size + 2*size_line_board+expand_size*2
    background_img = None
    if img_bg is not None:
        background_img = cv2.resize(img_bg,(w_img,h_img),interpolation=cv2.INTER_CUBIC)
    else:
        background_img = np.zeros(shape=[h_img, w_img, 3], dtype=np.uint8)
        background_img = np.full_like(background_img, fill_value=255)
    begin_point = None
    yb = int(expand_size + size_line_board / 2)
    ye = yb + max_size
    for i in range(len_final):
        xb = int(expand_size + size_line_board/2 + max_size*i)
        xe = xb + max_size
        cv2.rectangle(background_img,(xb,yb),(xe,ye),(10,23,13),int(size_line_board))
        xb_insert_img = int(xb + max_size/2 - list_bb_final[i][4]/2)
        yb_insert_img = int(yb + max_size/2 - list_bb_final[i][5]/2)
        crp_img_raw = crop_image(img, list_bb_final[i][0], list_bb_final[i][1], list_bb_final[i][2], list_bb_final[i][3])
        crp_img_bi  = invert_image(crp_img_raw)
        h_img_cr = crp_img_raw.shape[0]
        v_img_cr = crp_img_raw.shape[1]
        if color_text is None:
            for i in range(h_img_cr):
                for j in range(v_img_cr):
                    if crp_img_bi[i,j] > 0:
                        background_img[yb_insert_img + i,xb_insert_img + j] = crp_img_raw[i,j]
        else:
            for i in range(h_img_cr):
                for j in range(v_img_cr):
                    if crp_img_bi[i,j] > 0:
                        background_img[yb_insert_img + i,xb_insert_img + j] = color_text
    return background_img

def change_background_handwriting(img_bg,img,expand = 5,color_text = None):
    img2 = img.copy()
    h_img = img.shape[0]
    w_img = img.shape[1]
    img_bg_rs = cv2.resize(img_bg,(w_img,h_img),interpolation=cv2.INTER_CUBIC)
    print(img_bg_rs.shape)
    print(img2.shape)

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
    img_erase = erase_img(img, blank_image, expand)
    img_v = invert_image(img_erase)
    if color_text == None:
        for i in range(h_img):
            for j in range(w_img):
                if img_v[i,j] > 0:
                    img_bg_rs[i,j] = img2[i,j]
    else:
        for i in range(h_img):
            for j in range(w_img):
                if img_v[i,j] > 0:
                    img_bg_rs[i,j] = color_text
    return img_bg_rs


def crop_fit_text(img,expand = 5):
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
        if w/h < 3 and h/w < 3 and w > 10 and h > 10:
            if x < xtop_left:
                xtop_left = x
            if y < ytop_left:
                ytop_left = y
            if x + w > xbt_right:
                xbt_right = x+w
            if y+h > ybt_right:
                ybt_right = y+h
            cv2.rectangle(img2, (x , y), (x + w, y + h), (0, 255, 0), 1)
    img_crop = crop_image(img_erase, max(0,xtop_left - expand), max(0,ytop_left - expand),\
                          min(xbt_right + expand, w_img), min(ybt_right + expand, h_img))
    return img_crop

######################################## gen data script ##################################

def gen_data_number_with_box(path, path_save):
    list_img = ["3.jpg", "4.jpg", "5.jpg", "6.jpg", "10.jpg", "11.jpg", "12.jpg", "15.jpg",
                "16.jpg", "17.jpg", "18.jpg", "21.jpg", "22.jpg", "23.jpg", "24.jpg"]
    list_txt = ["3.txt", "4.txt", "5.txt", "6.txt", "10.txt", "11.txt", "12.txt", "15.txt",
                "16.txt", "17.txt", "18.txt", "21.txt", "22.txt", "23.txt", "24.txt"]
    list_path = os.listdir(path)
    for pathd in list_path:
        path_data = os.path.join(path,pathd)
        path_save_data = os.path.join(path_save,pathd)
        if not os.path.isdir(path_save_data):
            os.mkdir(path_save_data)
        for i in range(len(list_img)):
            gt_path = os.path.join(path_data,list_txt[i])
            img_pth = os.path.join(path_data,list_img[i])
            shutil.copy(gt_path,path_save_data)
            img = cv2.imread(img_pth)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            img = crop_fit_text(img)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            imgrs = aument_gen_list_box_number(img,expand_size_char=1)
            path_save_img = os.path.join(path_save_data,list_img[i])
            cv2.imwrite(path_save_img,imgrs)

def gen_data_path(path,path_save):
    list_img = list_files1(path,"jpg")
    list_img += list_files1(path,"png")
    dir_name = os.path.dirname(path)
    save_dir_line = os.path.join(path_save, dir_name+"_lines")

    if not os.path.isdir(save_dir_line):
        os.mkdir(save_dir_line)

    save_dir_dot = os.path.join(path_save, dir_name + "_dots")
    if not os.path.isdir(save_dir_dot):
        os.mkdir(save_dir_dot)

    save_dir_ldot = os.path.join(path_save, dir_name + "_linedots")
    if not os.path.isdir(save_dir_ldot):
        os.mkdir(save_dir_ldot)

    for img_path in list_img:
        path_img = os.path.join(path,img_path)
        print(path_img)
        img = cv2.imread(path_img)
        #line
        rs = augment_image_addline(img, type = 0)
        path_save = os.path.join(save_dir_line,img_path)
        cv2.imwrite(path_save,rs)
        # line
        rs = augment_image_addline(img, type = 1)
        path_save = os.path.join(save_dir_dot, img_path)
        cv2.imwrite(path_save, rs)
        # line
        rs = augment_image_addline(img, type = 2)
        path_save = os.path.join(save_dir_ldot, img_path)
        cv2.imwrite(path_save, rs)

def gen_white_data(path,path_save):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.basename(path)
    save_dir_white_line = os.path.join(path_save, 'add_solid_white_line', dir_name)

    if not os.path.isdir(save_dir_white_line):
        os.mkdir(save_dir_white_line)
    count = 0
    for img_name in list_img:
        path_img = os.path.join(path, img_name)
        count += 1
        print(count, path_img)
        img = cv2.imread(path_img)

        anno_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path_src = os.path.join(path, anno_name)
        # solid
        rs = augment_image_addline(img, 0, black=False)
        img_path_save = os.path.join(save_dir_white_line, img_name)
        anno_path_save = os.path.join(save_dir_white_line, anno_name)
        cv2.imwrite(img_path_save, rs)
        shutil.copy(anno_path_src, anno_path_save)

def gen_random_rotate(path,path_save):
    list_img = list_files1(path, "jpg")
    list_img += list_files1(path, "png")
    dir_name = os.path.basename(path)
    save_dir_rotate = os.path.join(path_save, 'add_rotate', dir_name)

    if not os.path.isdir(save_dir_rotate):
        os.mkdir(save_dir_rotate)

    count = 0
    for img_name in list_img:
        path_img = os.path.join(path, img_name)
        count += 1
        print(count, path_img)
        img = cv2.imread(path_img)

        anno_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path_src = os.path.join(path, anno_name)
        # solid
        rs = augment_random_rotate(img, -5,5)
        img_path_save = os.path.join(save_dir_rotate, img_name)
        anno_path_save = os.path.join(save_dir_rotate, anno_name)
        cv2.imwrite(img_path_save, rs)
        shutil.copy(anno_path_src, anno_path_save)

####################################### Class for pytorch #######################################
class cnx_aug_add_line(object):
    def __init__(self, color = (60, 60, 60),type = 0,size_draw  = 3, range_dot = 15,sizelinedot = 1):
    # type 0 solid line, 1 dots line,2 line dot
        self.type = type
        self.size_draw = size_draw
        self.range_dot = range_dot
        self.sizelinedot = sizelinedot
        self.color = color

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR)
        img_rs =  augment_image_addline(cv_img,self.color, self.type, self.size_draw, self.range_dot, self.sizelinedot)
        img_rs = cv2.cvtColor(img_rs,cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_autorotate(object):
     def __init__(self, begin_ag, end_ag, pixel_erase = 3):
         self.angle_bg = begin_ag
         self.angle_end = end_ag
         self.pixel_erase = pixel_erase

     def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_random_rotate(cv_img,self.angle_bg,self.angle_end, self.pixel_erase)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


class cnx_aug_random_erase(object):
    def __init__(self, numb_char_erase):
        self.numb_char_erase = numb_char_erase

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_random_erase(cv_img,self.numb_char_erase)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_bold_characters(object):
    def __init__(self,  kernel_dms = 3, iter = 1):
        self.kernel_dms = kernel_dms
        self.iter       = iter

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_bold_characters(cv_img,self.kernel_dms, self.iter)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_thin_characters(object):
    def __init__(self,  kernel_dms = 3, iter = 1):
        self.kernel_dms = kernel_dms
        self.iter       = iter

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_thin_characters(cv_img,self.kernel_dms, self.iter)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_blur(object):
    def __init__(self, type_blur = 0, size_window = 5):
        self.type_blur      = type_blur
        self.size_window    = size_window

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_blur(cv_img,self.type_blur, self.size_window)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_add_noise(object):
    def __init__(self):
        pass
    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_add_noise(cv_img)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)

class cnx_aug_resize_img(object):
    def __init__(self, w, h, fsx = None, fsy = None):
        self.w = w
        self.h = h
        self.fsx = fsx
        self.fsy = fsy

    def __call__(self, img):
        numpy_image = np.array(img)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        img_rs = augment_resize_img(cv_img, self.w, self.h, self.fsx, self.fsy)
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rs)


if __name__ == "__main__":

    transform_train = transforms.Compose([cnx_aug_resize_img(None,None,0.5,0.5)])
    # path = "C:/Users/chungnx/Desktop/aicr_data_hw/image_crop/aicrhw_2020-02-27_16-35/AICR_P0000035"
    path  = "C:/Users/chungnx/Desktop/image_digit_text"
    list_img = list_files1(path, "jpg")
    print(list_img)
    for l in list_img:
        path_img = os.path.join(path, l)
        img = cv2.imread(path_img)
        img = crop_fit_text(img)
        # imgrs = aument_gen_list_box_number(img)
        cv2.imshow("crop", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_rs = transform_train(img)
        numpy_image = np.array(img_rs)
        cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("rotate",cv_img)
        # cv2.imshow("rs",imgrs)
        cv2.waitKey()