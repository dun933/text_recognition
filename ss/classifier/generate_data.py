import os, glob
import json
from threading import Thread as worker
import cv2, codecs
from utils.utils import path_check, crop_from_img_square, crop_from_img_rectangle
from utils.self_augmentation import *
from config.config_manager import ConfigManager

os.environ["PYTHONIOENCODING"] = "utf-8"
configfile = 'config/classifier_config.ini'
configmanager = ConfigManager(configfile)

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names
# generate test data
data_dir = '/home/advlab/data/test_vn/Cello_data/'
img_dir = data_dir + '/test_image_vn/image_fix'
GT_dir = data_dir + '/test_image_vn/v7/ground_truth'
#img_dir =  data_dir+ 'imgs'
#GT_dir =  data_dir+ 'anno/v1'

file_list = get_list_file_in_folder(img_dir)
file_list = [x.replace('.png','').replace('.jpg','') for x in file_list]

# generate data for classifier
detection_data = "/home/advlab/data/corpus_250000_2019-12-12_19-28/"
annot_path = detection_data + "/annots"
img_path = detection_data + "/images"

# classifier 12
shape = 'rect' #rect #square
special_list='uUư.,-:oO0eE1tTiLlIvwWyYShnCca9gqSs'
merge_list=configmanager.ignore_class_list
num_sample_normal = 25000  # num sample of each normal char
num_sample_special = 50000  # num sample of char that more common
wh_ratio=2
rect_mode=1 # 3*w x h
num_class=228
with open("config/classmap_2stages_"+str(num_class)+".json", 'r') as f:
    loaded_json = json.load(f)
classifier12_list=[]
for cls in loaded_json.keys():
    classifier12_list.append(cls)

classifier12_output_dir = detection_data + "/classifier12_data_train_" + str(
    num_sample_normal) + '_' + str(num_sample_special) + '_'+ shape  + '_whratio_'+str(wh_ratio) +'_mode_'+str(rect_mode)+'_228_classes_test'
classifier12_test_dir = data_dir + 'anno/v1/classifier12_test_' + shape + '_whratio_'+str(wh_ratio) +'_mode_'+str(rect_mode)+'_228_classes_test'

# generate data for classifier 12
def gen_data_classifier12_from_detector(output_dir,input_list, out_shape='rect', remake=True):
    path_check(output_dir, remake=remake)
    annots = glob.glob(annot_path + "/*")
    class_dict = {}

    str_list = [x.replace('.', 'dot').replace('/', 'slash') for x in input_list]
    special_lis = [x.replace('.', 'dot').replace('/', 'slash') for x in special_list]
    count=0
    for sub_class in str_list:
        count+=1
        class_dict[sub_class] = 0
        path_check(os.path.join(output_dir, sub_class))
    print("Total class:",count)
    num_finished_class = 0
    # print(class_dict)
    for a_idx, annot in enumerate(annots):
        if (num_finished_class >= len(str_list)):
            continue
        filename = os.path.basename(annot).split(".")[0] + '.' + os.path.basename(annot).split(".")[1]+ '.' + os.path.basename(annot).split(".")[2]
        with open(annot) as json_file:
            json_data = json.load(json_file)
            img_file_name = img_path + "/" + filename.replace('.json', '.png') + '.png'
            img = cv2.imread(img_file_name)
            try:
                _h, _w, _ = img.shape
                for idx, v in enumerate(json_data['data']):
                    sub_class = v['char']
                    if sub_class in merge_list and num_class==201:  # merge lower and upper class to upper class here
                        sub_class = sub_class.upper()
                    if sub_class == '.':  # change . folder to dot folder
                        sub_class = 'dot'
                    if sub_class == '/':
                        sub_class = 'slash'  # change / folder to slash folder
                    if not sub_class in str_list:
                       continue
                    if class_dict[sub_class] >= num_sample_normal and sub_class not in special_lis:
                        continue
                    if class_dict[sub_class] >= num_sample_special and sub_class in special_lis:
                        continue
                    if class_dict[sub_class] % 1000 == 0 and class_dict[sub_class] > 0:
                        print('Class:', sub_class, "reach", str(class_dict[sub_class]))

                    # coords = [c for c in v['coords'].values()]
                    coords = [int(v['x1'] * _w), int(v['y1'] * _h), int(v['x2'] * _w), int(v['y2'] * _h)]
                    if out_shape == 'square':
                        roi = crop_from_img_square(img, coords)
                    elif out_shape == 'rect':
                        roi = crop_from_img_rectangle(img, coords, wh_ratio=wh_ratio, mode=rect_mode)
                    # print(roi.shape)
                    roi = total_augment(roi)
                    if roi is not None and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                        class_dict[sub_class] = class_dict[sub_class] + 1
                        if (class_dict[sub_class] >= num_sample_normal and sub_class not in special_lis) or \
                                (class_dict[sub_class] >= num_sample_special and sub_class in special_lis):
                            num_finished_class += 1
                            print(num_finished_class, 'Class:', sub_class,
                                  'get enough samples---------------------------------------------------')
                        save_as = "{}_{}.png".format(filename, str(idx).zfill(4))
                        cv2.imwrite(os.path.join(output_dir, sub_class, save_as), roi)
            except:
                print("No file:", img_file_name)
                pass

def generate_data_classifier12(out_shape, input_list):
    t = worker(target=gen_data_classifier12_from_detector, args=(classifier12_output_dir,input_list, out_shape, True))
    t.start()
    t.join()

def generate_test_data_classifier12_from_GT(out_shape='rect'):
    if not os.path.exists(classifier12_test_dir):
        os.makedirs(classifier12_test_dir)
    for sub_class in classifier12_list:
        print (sub_class)
        if not os.path.exists(os.path.join(classifier12_test_dir, sub_class)):
            os.makedirs(os.path.join(classifier12_test_dir, sub_class))

    count =0
    for file_name in file_list:
        ori_img_path = os.path.join(img_dir, file_name + '.png')
        GT_file = os.path.join(GT_dir, file_name + '.txt')
        print("create test data for image:", ori_img_path)
        if os.path.exists(os.path.join(img_dir, file_name + '.jpg')):
            ori_img_path = os.path.join(img_dir, file_name + '.jpg')
        fh1 = codecs.open(GT_file, "r", encoding='utf-8-sig')

        img = cv2.imread(ori_img_path)
        for idx, line in enumerate(fh1):
            data = (line.replace("\n", "")).split(" ")
            if data[0] in merge_list and num_class==201:  # merge lower and upper class to upper class here
                data[0] = data[0].upper()
            if data[0] == '.':  # change . folder to dot folder
                data[0] = 'dot'
            if data[0] == '/':  # change / folder to slash folder
                data[0] = 'slash'
            if data[0] in classifier12_list:
                xmin = int(data[1])
                ymin = int(data[2])
                width = int(data[3])
                height = int(data[4])
                coords = [xmin, ymin, xmin + width, ymin + height]
                if (out_shape == 'square'):
                    roi = crop_from_img_square(img, coords)
                if (out_shape == 'rect'):
                    roi = crop_from_img_rectangle(img, coords, wh_ratio=wh_ratio,  mode=rect_mode)
                save_as = "{}_{}.png".format(file_name, str(idx).zfill(4))

                cv2.imwrite(os.path.join(classifier12_test_dir, data[0], save_as), roi)
                count+=1
                print(count, 'Write char', data[0])

if __name__ == "__main__":
    generate_data_classifier12(out_shape=shape, input_list=classifier12_list)
    #generate_test_data_classifier12_from_GT(out_shape=shape)
    print("done.")
