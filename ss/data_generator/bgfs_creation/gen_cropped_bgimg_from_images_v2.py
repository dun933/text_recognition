"""
generated cropped image from single image depends on original image size
crops will retain original image zoom_aspect_ratio.
"""


import os, cv2, random, sys, math, datetime, math, argparse

parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)

def load_imgfilepaths(dirpath):
    file_list=[]
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            file_list.append(os.path.join(root, f))
    
    return file_list

def load_imgfilepaths_from_srcdirs(dirpath_list, ext_list):
    file_list=[]

    for dirpath in dirpath_list:
        file_list.extend(load_imgfilepaths(dirpath))
    filtered_output=[]
    for f in file_list:
        _, ext = os.path.splitext(f)
        ext = ext[1:]
        if ext in ext_list:
            filtered_output.append(f)
    return filtered_output


def random_gen_rect_coords(img_size, min_size):
    """
    img_size: (img_w, img_h) tuple in pixels
    min_size: (min_w, min_h) tuple in pixels
    """

    img_w, img_h = img_size

    if min_size is None:
        min_w = None
        min_h = None
    else:
        min_w, min_h = min_size
    
    if min_w is None:
        min_w =0
    if min_h is None:
        min_h =0

    w = random.randint(min_w, img_w)
    h = random.randint(min_h, img_h)

    max_x1_range = img_w - w
    max_y1_range = img_h - h

    x1 = random.randint(0, max_x1_range)
    y1 = random.randint(0, max_y1_range)

    x2 = x1 + w 
    y2 = y1+ h

    return x1,y1,x2,y2


def random_crop(imgmat, resize_size, min_w_percentage=0.1, min_h_percentage=0.1):
    img_h, img_w, _ = imgmat.shape
    img_size = (img_w, img_h)

    min_w = int( img_w * min_w_percentage)
    min_h = int (img_h * min_h_percentage)

    min_size = (min_w, min_h)

    x1,y1,x2,y2 = random_gen_rect_coords(img_size, min_size)
    
    roi = imgmat[y1:y2, x1:x2]

    resized = cv2.resize(roi, resize_size)
    return resized

def gen_crops(imgfilepath, crop_size=(320,320)):

    assert len(crop_size)==2
    assert isinstance(crop_size, tuple)

    # crop_w, crop_h = crop_size

    
    
    imgmat = cv2.imread(imgfilepath)

    imgmat = guarantee_min_size(imgmat, crop_size)


    img_h, img_w, _ = imgmat.shape
    img_size = (img_w, img_h)

    crop_loop_num = fetch_crop_number_based_on_img_size(img_size, crop_size)

    cropped_img_list= crop_img_with_loop(imgmat, crop_loop_num, crop_size)

    return cropped_img_list

def crop_img_with_loop(imgmat, crop_gen_num, crop_size):
    output=[]

    # calculate max zoomscale
    img_h, img_w, _ = imgmat.shape
    crop_w, crop_h = crop_size
    

    w_ratio = img_w / crop_w
    h_ratio = img_h /crop_h

    max_zoom_scale = min(w_ratio, h_ratio)

    for _ in range(crop_gen_num):
        output.append(crop_img(imgmat, max_zoom_scale,crop_size))

    return output

def crop_img(imgmat, max_zoom_scale, crop_size):

    img_h, img_w, _ = imgmat.shape
    crop_w, crop_h = crop_size

    img_size = (img_w, img_h)
    
    if max_zoom_scale< 1:
        raise Exception("invalid max_zoom_scale value={}".format(max_zoom_scale))
    elif max_zoom_scale==1.0:
        zoom_scale = 1.0
    else:
        zoom_scale = random_between_two_float(1.0, max_zoom_scale)

    actual_crop_w = int(crop_w * zoom_scale)
    actual_crop_h = int(crop_h * zoom_scale)

    rect_size = (actual_crop_w, actual_crop_h)

    x1,y1,x2, y2 = calc_random_rect_coords(img_size, rect_size)

    roi = imgmat[y1:y2, x1:x2]

    resized = cv2.resize(roi,crop_size, interpolation=cv2.INTER_CUBIC)

    return resized

def random_between_two_float(num1, num2):
    if num2 <= num1:
        raise Exception("num2 should be bigger than num1")
    
    random_float = random.random()

    delta = num2 - num1

    return num1 + random_float*delta



    
def calc_random_rect_coords(img_size, rect_size):
    rect_w, rect_h = rect_size
    img_w, img_h = img_size

    cx_min = math.ceil(rect_w/2)
    cx_max = img_w - math.ceil(rect_w/2)

    cy_min = math.ceil(rect_h/2)
    cy_max = img_h - math.ceil(rect_h/2)

    cx = random.randint(cx_min, cx_max)
    cy = random.randint(cy_min, cy_max)


    x1 = cx - rect_w/2
    x2 = cx + rect_w
    y1 = cy - rect_h/2
    y2 = y1 + rect_h

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    return x1,y1,x2,y2
        


def guarantee_min_size(imgmat, crop_size):
    img_h, img_w, _ = imgmat.shape

    crop_w, crop_h = crop_size

    w_ratio = float(img_w) / crop_w
    h_ratio = float(img_h) / crop_h

    min_dimension = min(w_ratio, h_ratio)

    if min_dimension >=1:
        return imgmat

    scale = 1/min_dimension
    
    resized = cv2.resize(imgmat, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    
    return resized

def fetch_crop_number_based_on_img_size(img_size, crop_size, numboost_factor=1):
    img_w, img_h = img_size
    crop_w, crop_h = crop_size

    img_area = img_w * img_h
    crop_area = crop_w * crop_h


    base_num = int(img_area/ crop_area) +1

    return base_num* numboost_factor


## MAIN

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-d", type=str, nargs=1,help="directory containing the raw background images")

    args = parser.parse_args()

    crop_size = (320, 320)
    if args.d is None:
        #raise Exception("-d option missing")
        src_img_dirs = ['/data/data_thang/gen_bg/']
        print("-d option missing")
    else:
        src_img_dirs = args.d

            
        
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    output_save_dir="cropped_bgs_{}".format(timestamp)

    print("src_img_dirs: {}".format(src_img_dirs))
    img_ext_filter=['png','jpg', 'jpeg']


    random.seed()
    output_save_dir = os.path.join(src_img_dirs[0], output_save_dir)
    try:
        os.makedirs(output_save_dir)
    except:
        print("make folder err")
        pass
    
    img_file_paths = load_imgfilepaths_from_srcdirs(src_img_dirs, img_ext_filter)

    # print(len(img_file_paths)) 

    # sys.exit(0)
    # load output image size

    count=0

    for f in img_file_paths:
        print("start processing {}".format(f))
        
        generated_crops = gen_crops(f, crop_size=crop_size)

        for croped_imgmat in generated_crops:

            output_filename="{}_{:07d}.png".format(timestamp, count)
            count+=1

            output_filepath = os.path.join(output_save_dir, output_filename)
            if not cv2.imwrite(output_filepath, croped_imgmat):
                print("save file false")
            # else:
            #     print("created {}".format(output_filepath))

    
    print("end of code")
