from fontTools.ttLib import TTFont
import random, cairocffi as cairo, cv2, sys, os
import functools, logging
import math, subprocess

from .language import Language
try:
    from CharBox import TextFailure, Box, CharBox, is_overlapping
    from cairo_fns import convert_cairo_format_to_rgb255base, convert_rgb255base_to_cairo_format, \
        convert_surface_to_cvmat, create_rgb24_cairosurface_from_cv2mat
    from image_augmentation_fns import apply_gaussian_blur, apply_salt_and_pepper, apply_low_res_by_resizing, \
        apply_invert, apply_jpeg_compression, AugmentationManager, PipeLineAugmentationManager
    from Charset import Charset

except ImportError:
    from .CharBox import TextFailure, Box, CharBox, is_overlapping
    from .cairo_fns import convert_cairo_format_to_rgb255base, convert_rgb255base_to_cairo_format, \
        convert_surface_to_cvmat, create_rgb24_cairosurface_from_cv2mat
    from .image_augmentation_fns import apply_gaussian_blur, apply_salt_and_pepper, apply_low_res_by_resizing, \
        apply_invert, apply_jpeg_compression, AugmentationManager, PipeLineAugmentationManager
    from .Charset import Charset

try:
    from BackgroundItem import BackgroundType
    from config.config_Vietnamese import Config_Vietnamese
    from config.ConfigManager import ConfigManager
except ImportError:
    from ..BackgroundItem import BackgroundType

import numpy as np

PACKAGE_PARENT = '../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(sys.path)
from dataprovision_utils.full_width_half_width import halfen, fullen, is_half_width, is_full_width

plaugmgr = PipeLineAugmentationManager()
plaugmgr.add_fn(apply_gaussian_blur, "apply_gaussian_blur", Config_Vietnamese.gauss_blur)
plaugmgr.add_fn(apply_salt_and_pepper, "apply_salt_and_pepper", Config_Vietnamese.salt_and_pepper)
plaugmgr.add_fn(apply_low_res_by_resizing, "ready_apply_low_res_by_resizing", Config_Vietnamese.low_res_by_resizing)
plaugmgr.add_fn(apply_invert, "ready_apply_invert", Config_Vietnamese.invert)
plaugmgr.add_fn(apply_jpeg_compression, "ready_apply_jpeg_compression", Config_Vietnamese.jpeg_compression)

dl = logging.getLogger("debug")

def get_sampled_colors(imgmat):
    #print('gen_image_fns.get_sampled_colors')
    img_h, img_w, _ = imgmat.shape
    sample_size = int((img_h * img_w) * 0.02)
    sample_size = max(sample_size, 10)
    sampled_colors = []
    for _ in range(sample_size):
        rand_x = random.randint(0, img_w - 1)
        rand_y = random.randint(0, img_h - 1)
        sampled_colors.append(imgmat[rand_y, rand_x])

    return sampled_colors

def get_random_ac_v2(sampled_colors, delta=50):
    #print('gen_image_fns.get_random_ac_v2')
    bgr_int_list = []

    for i in range(3):
        templist = []
        for j in range(256):
            templist.append(j)
        bgr_int_list.append(templist)

    for color in sampled_colors:
        for i in range(3):
            sc_component = color[i]
            min_range = max(0, sc_component - delta)
            max_range = min(255, sc_component + delta)

            for val in range(min_range, max_range + 1):
                if val in bgr_int_list[i]:
                    bgr_int_list[i].remove(val)

    # check if bgr_int_list is non-empty
    for i in range(3):
        list_size = len(bgr_int_list[i])
        # print("list_size={}".format(list_size))
        if list_size == 0:
            return None
    # random pick
    sel_color = []
    for i in range(3):
        bucket_list = bgr_int_list[i]
        picked_val = random.choice(bucket_list)
        sel_color.append(picked_val)

    return sel_color

def adjust_box_abscoords_for_high_aspect_ratio_char(targetchar, x1, y1, x2, y2, img_w, img_h):
    #print('gen_image_fns.adjust_box_abscoords_for_high_aspect_ratio_char')
    cx = float(x1 + x2) / 2
    cy = float(y1 + y2) / 2

    w = max(x2 - x1, 1)
    h = y2 - y1
    new_w = w * np.log(h / w)
    new_h = h

    new_x1 = cx - new_w / 2
    new_x2 = cx + new_w / 2

    new_y1 = cy - new_h / 2
    new_y2 = cy + new_h / 2

    # adjust oob
    new_x1 = max(new_x1,0)
    new_x2 = min(new_x2,img_w)

    new_y1 = max(new_y1,0)
    new_y2 = min(new_y2,img_h)
    return new_x1, new_y1, new_x2, new_y2

def adjust_box_abscoords_for_smspchar(targetchar, x1, y1, x2, y2, img_w, img_h, max_height):
    #print('gen_image_fns.adjust_box_abscoords_for_smspchar')
    cx = float(x1 + x2) / 2
    cy = float(y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1
    new_w = w
    new_h = h
    #'*:,@$.-(#%")/~!^&_+={}[]\;<>?※'
    if targetchar in ["."]:
        new_w = 1.5 * w
        new_h = 1.2 * w
    elif targetchar in ['*', '\"']:
        new_w = 1.5 * w
        new_h = 1.2 * h
    elif targetchar in ['^']:
        new_w = 1.3 * w
        new_h = 1.2 * h
    elif targetchar in [",", "'"]:
        new_w = 1.5 * w
        new_h = 1.2 * h
    elif targetchar in ["-", "_", "~"]:
        new_w = 1.2 * w
        new_h = 1.2 * h  # making the box square

    new_x1 = cx - new_w / 2
    new_x2 = cx + new_w / 2
    new_y1 = cy - new_h / 2
    new_y2 = cy + new_h / 2

    # adjust oob
    new_x1 = max(new_x1,0)
    new_x2 = min(new_x2,img_w)

    new_y1 = max(new_y1,0)
    new_y2 = min(new_y2,img_h)

    if targetchar in "\"^'":
        new_y2 = y1 + max_height * 0.8
    elif targetchar in "_,.":
        new_y1 = y2 - max_height * 0.8
    elif targetchar in "-~*=":
        new_y1 = cy - max_height * 0.4
        new_y2 = cy + max_height * 0.4

    return new_x1, new_y1, new_x2, new_y2

def generate_single_image_from_word_list_v1(**kwargs):
    #print('gen_image_fns.generate_single_image_from_word_list_v1')

    img_width = kwargs["img_width"]
    img_height = kwargs["img_height"]
    font_min_size = kwargs["font_min_size"]
    font_max_size = kwargs["font_max_size"]
    word_list = kwargs["word_list"]

    font_weight = kwargs["font_weight"]
    font_slant = kwargs["font_slant"]
    # font_basecolor = kwargs["font_basecolor"]
    # font_outlinecolor = kwargs["font_outlinecolor"] # not yet implemented
    font_typeface = kwargs["font_typeface"]
    font_AA_enable = kwargs["font_AA_enable"]

    bgitem = kwargs["bgitem"]
    remove_unknown = kwargs["remove_unknown"]
    remove_invisible = kwargs["remove_invisible"]
    char_stats_dict = kwargs["char_stats_dict"]
    unsupported_char = kwargs["unsupported_char"]
    charset = kwargs["charset"]
    if bgitem.type == BackgroundType.SOLIDCOLOR:
        raise Exception("cannot be solid color backgroundType")
    else:
        surface = cairo.ImageSurface.create_from_png(bgitem.imgfilepath)
        context = cairo.Context(surface)
        cv2imgmat = cv2.imread(bgitem.imgfilepath)

    box_list = []
    charbox_list = []

    context.select_font_face(font_typeface, font_slant, font_weight)
    kw = {
        "context": context,
        "font_min_size": font_min_size,
        "font_max_size": font_max_size,
        "word_list": word_list,
        "img_width": img_width,
        "img_height": img_height,
        "box_list": box_list,
        "charbox_list": charbox_list,
        "bgitem": bgitem,
        "cv2imgmat": cv2imgmat,
        "font_AA_enable": font_AA_enable,
        "remove_unknown": remove_unknown,
        "remove_invisible": remove_invisible,
        "char_stats_dict": char_stats_dict,
        "font_typeface": font_typeface,
        "unsupported_char": unsupported_char,
        "charset": charset
    }
    font_size = keep_drawing_words_on_surface(**kw)
    imgmat = convert_surface_to_cvmat(surface, surface.get_width(), surface.get_height())
    imgmat = plaugmgr.apply_augmentation(imgmat, font_size)

    #cv2.imwrite("result_img.png", imgmat)
    img_h, img_w, _ = imgmat.shape
    assert img_h == img_height
    assert img_w == img_width

    rgb_imgmat = cv2.cvtColor(imgmat, cv2.COLOR_BGR2RGB)
    return rgb_imgmat, charbox_list, font_size

#nq.cuong changes space ratio here, for corpus
def keep_drawing_words_on_surface(**kw):
    #print('gen_image_fns.keep_drawing_words_on_surface')
    context = kw.get("context", None)
    font_min_size = kw["font_min_size"]
    font_max_size = kw["font_max_size"]
    img_width = kw["img_width"]
    img_height = kw["img_height"]
    charbox_list = kw["charbox_list"]
    cv2imgmat = kw["cv2imgmat"]
    font_AA_enable = kw["font_AA_enable"]
    word_list = kw["word_list"]
    remove_unknown = kw['remove_unknown']
    font_typeface = kw["font_typeface"]
    unsupported_char = kw["unsupported_char"]
    charset = kw["charset"]

    font_size = random.randint(font_min_size, font_max_size)
    space_ratio = 1.0
    # select random font color
    x1 = 0
    y1 = 0
    x2 = int(img_width * 0.3)
    y2 = int(img_height * 0.3)

    roi_imgmat = cv2imgmat[y1:y2, x1:x2]
    for _ in range(100):
        sampled_colors = get_sampled_colors(roi_imgmat)
        font_color = get_random_ac_v2(sampled_colors, delta=100)
        if font_color is not None:
            break

    if font_color is None:
        font_color = [0, 0, 0]

    if np.random.RandomState().randint(10) == 0:
        font_color = convert_rgb255base_to_cairo_format(font_color)
    else:
        font_color = [0, 0, 0]

    context.set_source_rgb(*font_color)
    context.set_font_size(font_size)

    # set AA option
    if font_AA_enable is False:
        current_fontoption = context.get_font_options()
        current_fontoption.set_antialias(cairo.ANTIALIAS_NONE)
        context.set_font_options(current_fontoption)

    # # selecting first start position
    image_left_margin = 10
    image_top_margin = 10
    image_right_margin = 10
    image_bottom_margin = 10

    line_gap = np.random.RandomState().choice(list(range(5, 10)) + [20] + [30])

    # calculate space width
    x_bearing, y_bearing, width3, height, x_advance, y_advance = context.text_extents("a b")
    x_bearing, y_bearing, width2, height, x_advance, y_advance = context.text_extents("a")
    x_bearing, y_bearing, width1, height, x_advance, y_advance = context.text_extents("b")

    space_width = width3 - width2 - width1
    # print("space_width subtract calculated :{}".format(space_width))

    line_max_width = img_width - image_left_margin - image_right_margin - 50

    line_word_list, max_y_bearing, max_height, min_y_bearing = gen_line_candidate_info(context, word_list, line_max_width, space_width, charset)
    start_x = image_left_margin
    start_y = image_top_margin + max_y_bearing + random.randint(0, 30)

    start_y_max = img_height - image_bottom_margin

    ##draw grid background
    grid_image_left_margin = np.random.RandomState().randint(2, 12)
    grid_image_top_margin = np.random.RandomState().randint(2, 12)
    grid_image_right_margin = np.random.RandomState().randint(2, 12)
    grid_image_bottom_margin = np.random.RandomState().randint(2, 12)

    context.set_line_width(np.random.RandomState().randint(1, 5))
    context.move_to(grid_image_left_margin, grid_image_top_margin)
    context.line_to(grid_image_left_margin, img_height - grid_image_bottom_margin)
    context.line_to(img_width - grid_image_right_margin, img_height - grid_image_bottom_margin)
    context.line_to(img_width - grid_image_right_margin, grid_image_top_margin)
    context.line_to(grid_image_left_margin, grid_image_top_margin)
    context.stroke()

    while True:
        start_x = image_left_margin + random.randint(0, 30)
        end_x = img_width - image_right_margin + random.randint(0, 30)
        # now start drawing the words
        context.move_to(start_x, start_y)
        end_y_flag = False
        for index, word in enumerate(line_word_list):
            word = word.replace('\n','') #cuongnd remove \n
            if end_y_flag == True:
                break
            matrix = cairo.Matrix(xx=font_size, yx=0, xy=0, yy=font_size, x0=0, y0=0)
            word_char_list = list(word)
            if np.random.RandomState().randint(0, 2) == 0:
                new_width = np.random.RandomState().randint(int(font_size * 0.5), int(font_size * 1.4))
                matrix.xx = round(new_width)
            #Italic
            if np.random.RandomState().randint(0, 2) == 0:
                rand_xy = random.randint(-10, 5)
                matrix.xy = rand_xy

            context.set_font_matrix(matrix)
            # draw word char by char and save charbox info
            is_drawing = True
            for char in word_char_list:
                is_supported = True
                if ord(char) in unsupported_char[font_typeface]:
                    is_supported = False
                    if is_full_width(char):
                        char = halfen(char)
                        if not (ord(char) in unsupported_char[font_typeface]):
                            is_supported = True

                if is_supported is False:
                    #print("keep_drawing_words_on_surface.Font \"{}\" does not support char {}:{}".format(font_typeface, char, hex(ord(char))))
                    is_drawing = False
                    break
            if not is_drawing:
                #print('Stop draw word','\"'+word+'\"')
                continue
            for char in word_char_list:
                offx = start_x  # + x_bearing #+ width/2.0
                offy = start_y  # + y_bearing #+ height/2.0
                context.move_to(offx, offy)
                radian = random.randint(-175 * 3, 175 * 3) / 10000
                #context.rotate(radian)
                if np.random.RandomState().randint(0, 2) == 0:
                    rand_yx = random.randint(-3, 3)
                    matrix.yx = rand_yx
                    #context.set_font_matrix(matrix)
                x_bearing, y_bearing, width, height, x_advance, y_advance = context.text_extents(char)

                if end_x < start_x + x_advance:
                    end_y_flag = True
                    #context.rotate(-radian)
                    context.move_to(start_x, start_y)
                    break

                context.show_text(char)
                matrix.yx = 0
                context.set_font_matrix(matrix)
                #context.rotate(-radian)
                context.move_to(start_x, start_y)

                x1, x2, y1, y2 = get_new_position(start_x, start_y, x_bearing, y_bearing, width, height,
                                                  img_width, img_height, max_height, charset, char, radian)

                if x2 < 1 and y2 < 1:
                    if is_full_width(char):
                        char = halfen(char)
                    charbox = CharBox(x1, y1, x2, y2, char, charset.get(char))
                    charbox_list.append(charbox)

                # update start_x and start_y
                #nq.cuong updated, corrected position to reduce space character
                if random.randint(0, 1) == 0:
                    start_x += x_advance * space_ratio
                else:
                    start_x += x_advance * 1.1 *space_ratio

            if index < len(line_word_list):
                start_x += font_size/3
                context.move_to(start_x, start_y)
            # context.show_text(word)

        if np.random.RandomState().randint(0, 3) == 0:
            if np.random.RandomState().randint(0, 10) == 0:
                repeat = np.random.RandomState().randint(1, 3)
                dash1 = np.random.RandomState().randint(1, 10)
                dash2 = np.random.RandomState().randint(1, 10)
                space = np.random.RandomState().randint(1, 10)
                offset = np.random.RandomState().randint(0, 5)
                dash_type = [dash1, space] if repeat == 1 else [dash1, space, dash2, space]
                context.set_dash(dash_type, offset)

            context.set_line_width(np.random.RandomState().randint(1, 5))
            end_y = start_y + np.random.RandomState().randint(0, line_gap + 1)
            context.move_to(grid_image_left_margin, end_y)
            context.line_to(img_width - grid_image_right_margin, end_y)
            context.stroke()

        # search for next line
        start_y += max_height
        start_y += line_gap
        line_word_list, max_y_bearing, max_height, _ = gen_line_candidate_info(context, word_list, line_max_width,
                                                                               space_width, charset)
        next_start_y = start_y + max_height
        if next_start_y > start_y_max:
            break
    return font_size

def get_new_position(start_x, start_y, x_bearing, y_bearing, width, height,  img_width, img_height, max_height, charset, char, radian=0):
    #print('gen_image_fns.get_new_position')
    margin_width = round(max(height / 20, 1))
    margin_height = round(max(width / 20, 1))
    x1 = start_x + x_bearing  # - margin_width
    y1 = start_y + y_bearing  # - margin_height
    x2 = start_x + x_bearing + width  # + margin_width
    y2 = start_y + y_bearing + height  # + margin_height

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x1 = -width / 2.0
    y1 = -height / 2.0
    x2 = width / 2.0
    y2 = height / 2.0

    rx1 = cx + (math.cos(radian) * x1 - math.sin(radian) * y1)
    ry1 = cy + (math.sin(radian) * x2 + math.cos(radian) * y1)
    rx2 = cx + (math.cos(radian) * x2 - math.sin(radian) * y2)
    ry2 = cy + (math.sin(radian) * x1 + math.cos(radian) * y2)

    x1, y1, x2, y2 = round(rx1), round(ry1), round(rx2), round(ry2)

    x1 -= margin_width
    y1 -= margin_height
    x2 += margin_width
    y2 += margin_height

    aspect_ratio = (height / (width + 1e-10))
    if aspect_ratio > 5.0:
        x1, y1, x2, y2 = adjust_box_abscoords_for_high_aspect_ratio_char(char, x1, y1, x2, y2, img_width,
                                                                         img_height)
    if charset.get(char) == "symbol":
        x1, y1, x2, y2 = adjust_box_abscoords_for_smspchar(char, x1, y1, x2, y2, img_width, img_height,
                                                           max_height)

    x1 = float(x1) / img_width
    x2 = float(x2) / img_width
    y1 = float(y1) / img_height
    y2 = float(y2) / img_height

    return x1, x2, y1, y2

def is_visible_char(char, **kwargs):
    #print('gen_image_fns.is_visible_char:',char)
    if 'font_path' in kwargs.keys():
        font_path = kwargs['font_path']
    elif 'font_name' in kwargs.keys():
        font_name = kwargs['font_name']
        font_path = get_font_path(font_name, Config_Vietnamese.font_dir)
        if font_path is None or len(font_path) < 4:
            print("Font path error: {}".format(font_name))
            return False

        if font_path[-3:] not in ['ttf', 'TTF']:
            font = TTFont(font_path, fontNumber=0)
        else:
            font = TTFont(font_path)
    elif 'font' in kwargs.keys():
        font = kwargs['font']
    else:
        raise Exception("missing argument")
    return has_glyph(font, char)

def has_glyph(font, glyph):
    #print('gen_image_fns.has_glyph:',font,', glyph:',glyph)
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

#cuongnd get font from dir for Window
def get_list_font_from_dir(font_dir, gen_more_popular_font=True):
    print('gen_image_fns.get_list_font_from_dir:',font_dir)
    list_font = list()
    for (dirpath, dirnames, filenames) in os.walk(font_dir):
        for file in filenames:
            list_font += [file.split('.')[0]]
    if(gen_more_popular_font==True):
        for i in range(10):
            list_font+=['Times_New_Roman']
    list_font=sorted(list_font)
    print('Font list:')
    for font in list_font:
        print(font)
    return list_font

def get_font_path(font_name, font_dir=''):
    #print('gen_image_fns.get_font_path:',font_name,', in dir:',font_dir)
    if(font_dir==''): #linux
        p = subprocess.run(['fc-list', font_name], stdout=subprocess.PIPE)
        output = p.stdout
        output = output.decode('utf-8')
        lines = output.split("\n")
        for line in lines:
            if len(line) == 0:
                continue
            splitted = line.split(":")
            return splitted[0]
    else:
        list_font_path = list()
        for (dirpath, dirnames, filenames) in os.walk(font_dir):
            list_font_path += [os.path.join(dirpath, file) for file in filenames]
        for font_path in list_font_path:
            font_name_from_path=os.path.basename(font_path).split('.')[0]
            if(font_name_from_path.lower()==font_name.lower()):
                #print('found font:',font_name,', in file:',font_path)
                return font_path
    return ''

def make_full_width(c, language, full_width_ratio=0.5):
    #print('gen_image_fns.make_full_width')
    if language is Language.CHINESE and is_half_width(c):
        tmp_rate = np.random.RandomState().random()
        if tmp_rate < full_width_ratio:
            c = fullen(c)
    return c

def gen_line_candidate_info(context, word_list, line_width, space_width, charset=None):
    #print('gen_image_fns.gen_line_candidate_info')
    line_word_list = []
    y_bearing_list = []
    word_height_list = []
    remain_line_width = line_width

    try_count = 0
    max_try_word_list = 5
    min_length = 2
    max_length = 6

    while True:
        count_down = try_count - max_try_word_list
        if count_down > 0:
            max_length_try = max_length - count_down + 1
            if max_length_try < 1:
                break
            min_length_try = min(max_length_try, min_length)
            #word = gen_random_word(min_length_try, max_length_try, charset)
        else:
            word = random.choice(word_list) #cuongnd no need to random word here anymore

        x_bearing, y_bearing, width, height, x_advance, y_advance = context.text_extents(word)

        if width > remain_line_width:
            try_count += 1
            continue
        else:
            try_count = 0

        remain_line_width = remain_line_width - width

        line_word_list.append(word)
        y_bearing_list.append(y_bearing)
        word_height_list.append(height)

        if remain_line_width > (space_width * 3):
            remain_line_width = remain_line_width - space_width
        else:
            # there is little space left to write anything... just end the line here.
            break
    if len(line_word_list) > 0:
        max_y_bearing = min(y_bearing_list)  # using min since y_bearaing should be negative values
        max_y_bearing = abs(max_y_bearing)
        min_y_bearing = min(y_bearing_list)
        max_height = max(word_height_list)
    else:
        max_y_bearing = 0
        min_y_bearing = 0
        max_height = 0

    return line_word_list, max_y_bearing, max_height, min_y_bearing

def get_statistics_for_font(char_list, font_type_face, font_size, slant=0, weight=0):
    #print('gen_image_fns.get_statistics_for_font:',font_type_face,', size:',font_size)
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 320, 320)
    context = cairo.Context(surface)
    context.paint()

    context.select_font_face(font_type_face, slant, weight)
    context.set_font_size(font_size)

    widths = []
    heights = []
    x_bearings = []
    y_bearings = []

    for c in char_list:
        x_bearing, y_bearing, width, height, x_advance, y_advance = context.text_extents(c)
        heights.append(height)
        widths.append(width)
        x_bearings.append(x_bearing)
        y_bearings.append(y_bearing)

    return np.mean(x_bearings), np.std(x_bearings), np.mean(y_bearings), np.std(y_bearings), \
           np.mean(widths), np.std(widths), np.mean(heights), np.std(heights)
