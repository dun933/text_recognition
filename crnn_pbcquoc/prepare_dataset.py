import os, cv2, random
import time

icdar_dir = '/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/outputs/corpus_100000_2020-01-31_14-26/images'
output_dir = '/home/duycuong/PycharmProjects/dataset/aicr_icdar_new'


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['jpg', 'png', 'JPG', 'PNG']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def crop_from_img_rectangle(img, left, top, right, bottom):
    extend_y = max(int((bottom - top) / 3), 4)
    extend_x = int(extend_y / 2)
    top = max(0, top - extend_y)
    bottom = min(img.shape[0], bottom + extend_y)
    left = max(0, left - extend_x)
    right = min(img.shape[1], right + extend_x)
    if left >= right or top >= bottom or left < 0 or right < 0 or left >= img.shape[1] or right >= img.shape[1]:
        return None
    return img[top:bottom, left:right]

def prepare_train_from_icdar(data_dir, output_dir):
    try:
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'annos'))
    except:
        pass

    list_files = get_list_file_in_folder(data_dir)
    train_txt = ''
    test_txt = ''
    max_wh = 0
    count = 0
    for idx, file in enumerate(list_files):
        # if(idx>1000):
        #     continue
        print(idx, file)
        base_name = file.replace('.png', '')
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)
        anno_path = img_path.replace('.png', '.txt')
        with open(anno_path, 'r', encoding='utf-8') as f:
            anno_list = f.readlines()
        anno_list = [x.strip() for x in anno_list]
        for index, anno in enumerate(anno_list):
            pts = anno.split(',')
            left = int(pts[0])
            top = int(pts[1])
            right = int(pts[2])
            bottom = int(pts[5])
            loc = -1
            for i in range(0, 8):
                loc = anno.find(',', loc + 1)
            val = anno[loc + 1:]
            crop = crop_from_img_rectangle(img, left, top, right, bottom)
            if crop is None or val == '':
                continue
            if ((crop.shape[1] / crop.shape[0]) > max_wh):
                max_wh = crop.shape[1] / crop.shape[0]
            count += 1
            base_crop_name = base_name + '_' + str(index)
            if (random.randint(0, 4) == 0):
                test_txt += 'images/' + base_crop_name + '.jpg\n'
            else:
                train_txt += 'images/' + base_crop_name + '.jpg\n'
            crop_img_path = os.path.join(output_dir, 'images', base_crop_name + '.jpg')
            crop_anno_path = os.path.join(output_dir, 'annos', base_crop_name + '.txt')
            cv2.imwrite(crop_img_path, crop)
            with open(crop_anno_path, 'w', encoding='utf-8') as f:
                f.write(val)

    with open(os.path.join(output_dir, 'train'), 'w', encoding='utf-8') as f:
        f.write(train_txt)
    with open(os.path.join(output_dir, 'test'), 'w', encoding='utf-8') as f:
        f.write(test_txt)
    print('max width height ratio in dataset', max_wh)
    print('Total word:', count)

def prepare_train_test_from_multiple_dir(root_dir, list_dir, percentage=1.0, train_ratio=0.9, convert=False):
    list_dir_txt = []
    print('root dir:', root_dir)
    from unicode_utils import compound_unicode
    for dir in list_dir:
        if os.path.exists(os.path.join(root_dir,dir,'origine.jpg')):
            os.remove(os.path.join(root_dir,dir,'origine.jpg'))
            print('remove file',os.path.join(root_dir,dir,'origine.jpg'))
        list_files = get_list_file_in_folder(os.path.join(root_dir, dir))
        #convert_json_to_multiple_gt(os.path.join(root_dir, dir))
        print('Dir ', dir, 'has', len(list_files), 'files')
        for idx, file in enumerate(list_files):
            print(file)
            list_dir_txt.append(os.path.join(dir, file))

            #fix unicode
            txt_file=os.path.join(root_dir,dir, file).replace('.jpg','.txt')
            with open(txt_file, 'r', encoding='utf-8') as f:
                txt= f.readlines()
            if len(txt)>0:
                new_text=compound_unicode(txt[0])
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_text)


    random.shuffle(list_dir_txt)
    print('\ntotal files:', len(list_dir_txt))
    num_files_to_use = int(percentage * len(list_dir_txt))
    num_files_to_train = int(train_ratio * num_files_to_use)
    num_files_to_test = num_files_to_use - num_files_to_train
    print('\ntrain files:', num_files_to_train)
    print('\ntest files:', num_files_to_test)

    train_list = list_dir_txt[0:num_files_to_train - 1]
    test_list = list_dir_txt[num_files_to_train:num_files_to_use - 1]

    print('Write train test file')
    train_txt = ''
    for train_file in train_list:
        train_txt += train_file + '\n'
    with open(os.path.join(root_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write(train_txt)

    test_txt = ''
    for test_file in test_list:
        test_txt += test_file + '\n'
    with open(os.path.join(root_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write(test_txt)

    print('Done')

def prepare_txt_file(data_dir):
    list_files = get_list_file_in_folder(data_dir)
    save_txt=''
    for file in list_files:
        save_txt+=os.path.join(data_dir,file)+'\n'
        with open(os.path.join(data_dir,file.replace('.jpg','.txt').replace('.png','.txt')), 'w', encoding='utf-8') as f:
            f.write('abc')
    with open(os.path.join(data_dir+'/..', 'test'), 'w', encoding='utf-8') as f:
        f.write(save_txt)

def convert_json_to_multiple_gt(dir, json_name='labels.json'):
    import json
    with open(os.path.join(dir, json_name)) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            gt_name = key.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(os.path.join(dir, gt_name), 'w', encoding='utf-8') as f:
                f.write(value)

def crop_collected_data(dir, file_list=['2','8','14','20'], debug=False):
    list_files = get_list_file_in_folder(dir)
    for file in list_files:
        if file.replace('.jpg','') in file_list:
            file_path=os.path.join(dir,file)
            print(file_path)
            ori_img= cv2.imread(file_path)
            crop_img=ori_img[0:82,0:ori_img.shape[1]]
            if debug:
                cv2.imshow('result',crop_img)
                cv2.waitKey(0)
            cv2.imwrite(file_path, crop_img)

def temp():
    import json
    with open('temp.json', 'w', encoding='utf-8') as f:
        json.dump('Đường Nguyễn Phong Sắc, Huyện Thủy Nguyên, Hải Phòng, Đường Lâm Hạ, Quận Long Biên, Hà Nội,  ̀', f,
                  ensure_ascii=True)

def crop_collected_data2(dir, debug=False):
    list_files = get_list_file_in_folder(dir)
    count1=0
    count2=0
    for file in list_files:
        file_path = os.path.join(dir, file)
        print(file_path)
        ori_img = cv2.imread(file_path)
        if ori_img.shape[1]/ori_img.shape[0]>9:
            count1+=1
            extend_val=  int(ori_img.shape[0]/9)+3
            print ('count1',count1,'extend val',extend_val,'height',ori_img.shape[0])
            crop_img = ori_img[extend_val:ori_img.shape[0] - extend_val - 1, 0:ori_img.shape[1]]
            if debug:
                cv2.imshow('ttt', crop_img)
                cv2.waitKey(0)
            cv2.imwrite(file_path, crop_img)
        elif ori_img.shape[1]/ori_img.shape[0]>7:
            count2+=1
            extend_val=  int(ori_img.shape[0]/9)+1
            print ('count2',count2,'extend val',extend_val,'height',ori_img.shape[0])
            crop_img = ori_img[extend_val:ori_img.shape[0] - extend_val - 1, 0:ori_img.shape[1]]
            cv2.imwrite(file_path, crop_img)
        if debug:
            cv2.imshow('result', ori_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # prepare_train_from_icdar(icdar_dir, output_dir)
    root_dir = '/home/duycuong/PycharmProjects/dataset/cleaned_data_merge_fixed'
    list_dir=get_list_dir_in_folder(root_dir)
    list_dir = sorted(list_dir)
    #list_dir = ['0825_DataSamples', '0916_DataSamples', '1015_Private Test','0825_DataSamples_dots', '0916_DataSamples_dots', '1015_Private Test_dots','0825_DataSamples_linedots', '0916_DataSamples_linedots', '1015_Private Test_linedots','0825_DataSamples_lines', '0916_DataSamples_lines', '1015_Private Test_lines']
    #prepare_train_test_from_multiple_dir(root_dir, list_dir)
    #prepare_txt_file('/home/duycuong/PycharmProjects/research_py3/text_recognition/EAST_argman/outputs/predict_handwriting_model.ckpt-45451/trang_new')
    # prepare_train_from_icdar(icdar_dir, output_dir)
    img_dir='/home/duycuong/PycharmProjects/dataset/ocr_dataset/meta'
    crop_collected_data2(img_dir)
    # for dir in list_dir:
    #     crop_collected_data(os.path.join(root_dir,dir))
