import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont, ImageDraw
from random import randint
import cv2, hgtk, os, random, glob, json
from datetime import datetime
from collections import OrderedDict
import shutil
import math

from pygments.lexer import combined

try:
    from utils.self_augmentation import *
except ImportError:
    from classifier.utils.self_augmentation import *

now = datetime.now()
padding=20

def path_check(path, remake=False):
    if os.path.exists(path) and remake:
        import shutil
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)

def image_generate(chs, bg_files, FONTS, out_path, canvas_size):
    global font, image, color, draw
    print("image gen start!@")
    img_out_path = out_path + "/images"
    anno_out_path = out_path + "/annots"
    path_check(img_out_path, True)
    path_check(anno_out_path, True)

    st_x = 20
    st_y = 20
    max_y2 = -1
    idx = 0
    L = len(chs)
    coords = []

    re_canvas_flag = True
    gen_imgs = []
    annots = OrderedDict()
    file_cnt = 0
    while idx < L:
        if idx % 1000 == 0:
            print("[character  {} / {}  is processed ... ]".format(idx, L))
        if re_canvas_flag:
            image = Image.open(np.random.choice(bg_files))  # 배경 위치
            # image = image.resize((320, 320))
            draw = ImageDraw.Draw(image)
            f_ = np.random.choice(FONTS)
            # print(f_)
            font = ImageFont.truetype(f_, random.randint(25, 60))  # 폰트사이즈
            color = (random.randint(0, 130), random.randint(0, 130), random.randint(0, 130), 255)  # 글자 색
            re_canvas_flag = False

        ch = chs[idx]
        W, H = font.getsize(ch)
        if ch != ' ':
            a,b,c,d = font.getmask(ch).getbbox()
            # print(a,b,c,d)
            Woff = c
            Hoff = d
        x, y = st_x, st_y
        x2, y2 = x + W, y + H
        # x = x2-Woff
        y = y2-Hoff

        max_y2 = max(max_y2, y2)

        if x2 > canvas_size[0]-40:
            y_gap = random.randint(10, 30)
            st_y = max_y2 + y_gap
            st_x = random.randint(10, 20)
            # st_x = 0
            continue
        elif max_y2 >= canvas_size[1]-40:
            re_canvas_flag = True
            st_x = 20
            st_y = 20
            max_y2 = -1

            res = cv2.cvtColor(np.array(image, np.uint8), cv2.COLOR_BGR2GRAY)
            # res = total_augment(res)
            ksize =  np.random.choice([3,5])
            res = blur(res, ksize=ksize)
            # print(annots)
            # draw bounding box for character
            for num,orddict in annots.items():
                # print(orddict['coords'])
                _x=orddict['coords']['x']
                _y = orddict['coords']['y']
                _x2 = orddict['coords']['x2']
                _y2 = orddict['coords']['y2']
                cv2.rectangle(res, (_x, _y), (_x2, _y2), 50, 1)

            filename = "{}_{}".format(str(file_cnt).zfill(6), '%s-%s-%s' % (now.year, now.month, now.day))
            cv2.imwrite(img_out_path + "/" + filename + ".png", res)

            gen_imgs.append(res)
            with open(anno_out_path + "/" + filename + ".json", 'w', encoding='utf-8') as mf:
                json.dump(annots, mf, ensure_ascii=False, indent='\t')

            annots = OrderedDict()
            file_cnt += 1
            continue
        # value

        obj_dict = OrderedDict()
        obj_dict["char"] = ch
        obj_dict["coords"] = {'x': x, 'y': y, 'x2': x2, 'y2': y2}

        annots[idx] = obj_dict

        draw.text((st_x, st_y), ch, font=font, fill=color)
        x_gap = random.randint(-1, 3)
        st_x = x2 + x_gap
        idx += 1
    print("generate done.")


def image_generate_for_testsfont(chs, bg_files, FONTS, out_path, canvas_size):
    print("image gen start!@")
    img_out_path = out_path + "/images"
    anno_out_path = out_path + "/annots"
    path_check(img_out_path, True)
    path_check(anno_out_path, True)

    for fff in FONTS:
        st_x = 20
        st_y = 20
        max_y2 = -1
        idx = 0
        L = len(chs)
        print(idx, L)
        file_cnt = 0
        image = Image.open(np.random.choice(bg_files))  # 배경 위치
        image = image.resize((320, 320))
        draw = ImageDraw.Draw(image)
        while idx < L:
            f_ = fff
            font = ImageFont.truetype(f_, random.randint(30, 45))  # 폰트사이즈
            # color = (random.randint(30, 170), random.randint(30, 170), random.randint(30, 170), 255) # 글자 색
            color = 0, 0, 0

            ch = chs[idx]
            W, H = font.getsize(ch)
            x, y = st_x, st_y
            x2, y2 = x + W, y + H
            max_y2 = max(max_y2, y2)

            if x2 > canvas_size[0]:
                y_gap = random.randint(10, 30)
                st_y = max_y2 + y_gap
                st_x = random.randint(10, 20)
                continue
            draw.text((st_x, st_y), ch, font=font, fill=color)
            x_gap = random.randint(0, 10)
            st_x = x2 + x_gap
            idx += 1

        res = cv2.cvtColor(np.array(image, np.uint8), cv2.COLOR_BGR2GRAY)
        filename = "{}_{}.png".format(os.path.basename(fff), 0)
        cv2.imwrite(img_out_path + "/" + filename, res)
    print("generate done.")


def model_path_setting(project_dir: str, model_name: str, remake=False):
    checkpoints_dir = project_dir + "/" + 'checkpoints'

    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    checkpoints_save_path = checkpoints_dir + "/" + model_name
    if not os.path.isdir(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)
    else:
        if remake:
            shutil.rmtree(checkpoints_save_path)
            os.mkdir(checkpoints_save_path)


def flow_from_directory_for_labels(path: str, sort: bool = True) -> dict:
    label_dict = dict()
    import glob
    dirs = glob.glob(path + "/*")
    dirs = sorted([x for x in dirs if os.path.isdir(x)])
    for idx, dir in enumerate(dirs):
        label_dict[idx] = os.path.basename(dir)
    return label_dict

def make_square(x, y, x2, y2):
    h = y2 - y
    w = x2 - x
    diff = h - w
    if diff < 0:
        y -= abs(int(diff / 2))
        y2 = y + w
    else:
        x -= abs(int(diff / 2))
        x2 = x + h
    return x, y, x2, y2

def make_rectangle(x, y, x2, y2, wh_ratio=2, mode=1):
    #mode 1: new width scale by height -> preserved scale when training
    #mode 2: new width scale by width -> stretched when training
    h = y2 - y
    w = x2 - x
    new_w=wh_ratio*h
    if(mode==2):
        new_w = wh_ratio*w
    extend_val = new_w - w
    if extend_val < 0:
        y -= abs(int(extend_val / 2))
        y2 = y + w
    else:
        x -= abs(int(extend_val / 2))
        x2 = x + new_w
    return x, y, x2, y2

def crop_from_img_square(img, coords):
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    x, y, x2, y2 = [c + padding for c in coords]
    h = y2 - y
    extend_val = max(int(h / 8),2)
    extend_y = y - extend_val
    extend_y2 = y2 + extend_val
    new_x, new_y, new_x2, new_y2 = make_square(x, extend_y, x2, extend_y2)

    re_x = max(0, new_x)
    re_y = max(0, new_y)
    re_x2 = min(img.shape[1], new_x2)
    re_y2 = min(img.shape[0], new_y2)
    return img[re_y:re_y2, re_x:re_x2]


def crop_from_img_rectangle(img, coords, wh_ratio=2, mode=1):  # wh_ratio=width/height
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    x, y, x2, y2 = [c + padding for c in coords]
    h = y2 - y
    extend_val = max(int(h / 8), 2) #big improvement
    extend_y = y - extend_val
    extend_y2 = y2 + extend_val
    new_x, new_y, new_x2, new_y2 = make_rectangle(x, extend_y, x2, extend_y2, wh_ratio, mode=mode)

    re_x = max(0, new_x)
    re_y = max(0, new_y)
    re_x2 = min(img.shape[1], new_x2)
    re_y2 = min(img.shape[0], new_y2)
    return img[re_y:re_y2, re_x:re_x2]


def crop_from_img_by_margin(org_img, coords, margin, background=None):
    if background is None:
        background = getBackground(org_img)

    x, y, x2, y2 = coords

    w = x2 - x
    h = y2 - y

    x_new = int(x - w*margin)
    x2_new = int(x2 + w*margin)
    y_new = int(y - h*margin)
    y2_new = int(y2 + h*margin)

    top = 0
    bottom = 0
    left = 0
    right = 0

    if x_new < 0:
        left = - x_new
        x_new = 0

    if x2_new > org_img.shape[1]:
        right = org_img.shape[1] - x2_new
        x2_new = org_img.shape[1]

    if y_new < 0:
        top = - y_new
        y_new = 0

    if y2_new > org_img.shape[0]:
        bottom = y2_new - org_img.shape[0]
        y2_new = org_img.shape[0]

    img = org_img[y_new:y2_new, x_new:x2_new]

    return cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=background)


def getBackground(img):
    height = img.shape[0]
    width = img.shape[1]

    margin = max(round(min(height, width) * 0.05) + 1, 2)

    channel = len(img.shape)
    if channel > 2:
        sum = np.zeros(channel)
    else:
        sum = 0
    count = 0
    for i in range(margin):
        for j in range(margin):
            sum += img[i][j]
            count += 1

    sum /= count
    # channel = len(img.shape)
    # if channel > 2:
    #     channel = img.shape[2]
    # else:
    #     channel = 1
    #
    # count = 0
    # if channel > 1:
    #     ret = np.zeros(channel)
    #     for i in range(margin):
    #         for j in range(margin):
    #             is_pass = True
    #             for k in range(channel):
    #                 if sum[i][j][k] < img[i][j][k]:
    #                     is_pass = False
    #                     break
    #             if is_pass:
    #                 ret += img[i][j]
    #                 count += 1
    # else:
    #     ret = 0
    #     for i in range(margin):
    #         for j in range(margin):
    #             if sum[i][j] < img[i][j]:
    #                 continue
    #
    #             ret += img[i][j]
    #             count += 1
    #
    # ret /= count

    return sum


def crop_from_img_by_margin_and_fill(org_img, coords, pattern_size, margin=0.05, background=None):
    if background is None:
        background = getBackground(org_img)
    img = crop_from_img_by_margin(org_img, coords, margin, background)
    return resize_keep_aspect_ratio_and_fill(img, pattern_size, background)


def resize_keep_aspect_ratio_and_fill(img, pattern_size, background=None):
    if background is None:
        background = getBackground(img)

    h = img.shape[0]
    w = img.shape[1]
    h_p = pattern_size[0]
    w_p = pattern_size[1]
    # print(h, w, h_p, w_p)
    aspect_ratio = float(h) / w
    aspect_ratio_p = float(h_p) / w_p
    top = 0
    bottom = 0
    left = 0
    right = 0
    if aspect_ratio_p > aspect_ratio:
        h_new = math.floor(aspect_ratio * w_p)
        img = cv2.resize(img, (w_p, h_new), interpolation=cv2.INTER_NEAREST)
        h_padding = math.floor((h_p - h_new) / 2)
        top = h_padding
        bottom = h_p - h_new - top
    else:
        w_new = math.floor(h_p / aspect_ratio)
        img = cv2.resize(img, (w_new, h_p), interpolation=cv2.INTER_NEAREST)
        w_pad = math.floor((w_p - w_new) / 2)
        left = w_pad
        right = w_p - w_new - left

    # print("something wrong ", img.shape, top, bottom, left, right)
    return cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=background)
