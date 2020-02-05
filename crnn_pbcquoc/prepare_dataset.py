
import os, cv2, random
import time


icdar_dir='/home/duycuong/PycharmProjects/research_py3/text_recognition/ss/data_generator/outputs/corpus_100000_2020-01-31_14-26/images'
output_dir='/home/duycuong/PycharmProjects/dataset/aicr_icdar_new'

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def crop_from_img_rectangle(img, left, top, right, bottom):
    extend_y= max(int((bottom-top) / 3), 4)
    extend_x=int(extend_y/2)
    top = max(0, top-extend_y)
    bottom = min(img.shape[0], bottom+extend_y)
    left = max(0, left-extend_x)
    right = min(img.shape[1], right+extend_x)
    if left>=right or top>=bottom or left<0 or right<0 or left >=img.shape[1] or right >=img.shape[1]:
        return None
    return img[top:bottom, left:right]

def prepare_train_from_icdar(data_dir, output_dir):
    try:
        os.makedirs(os.path.join(output_dir,'images'))
        os.makedirs(os.path.join(output_dir,'annos'))
    except:
        pass

    list_files=get_list_file_in_folder(data_dir)
    train_txt=''
    test_txt=''
    max_wh=0
    count=0
    for idx, file in enumerate(list_files):
        # if(idx>1000):
        #     continue
        print(idx, file)
        base_name=file.replace('.png', '')
        img_path=os.path.join(data_dir,file)
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
            if crop is None or val=='':
                continue
            if((crop.shape[1]/crop.shape[0])>max_wh):
                max_wh=crop.shape[1]/crop.shape[0]
            count+=1
            base_crop_name = base_name + '_' + str(index)
            if(random.randint(0,4)==0):
                test_txt += 'images/' + base_crop_name + '.jpg\n'
            else:
                train_txt += 'images/' + base_crop_name + '.jpg\n'
            crop_img_path = os.path.join(output_dir,'images', base_crop_name + '.jpg')
            crop_anno_path = os.path.join(output_dir,'annos', base_crop_name + '.txt')
            cv2.imwrite(crop_img_path, crop)
            with open(crop_anno_path, 'w', encoding='utf-8') as f:
                f.write(val)

    with open(os.path.join(output_dir,'train'), 'w', encoding='utf-8') as f:
        f.write(train_txt)
    with open(os.path.join(output_dir,'test'), 'w', encoding='utf-8') as f:
        f.write(test_txt)
    print('max width height ratio in dataset',max_wh)
    print('Total word:',count)

if __name__ == "__main__":
    prepare_train_from_icdar(icdar_dir, output_dir)