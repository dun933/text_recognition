import torch
from torch.autograd import Variable
from models.utils import strLabelConverter
import models.crnn as crnn
import time, os
import pickle

import cv2
from utils.loader import alignCollate, NumpyListLoader
import models.utils as utils

import config
from torchvision import transforms
from pre_processing.image_preprocessing import extract_for_demo
from pre_processing.image_calibration import calib_image
from symspellpy.address_spell_check import correct_address
from symspellpy.name_spell_check import load_name_corection, correct_name
from symspellpy.country_spell_check import load_country_correction, correct_country
from form.form_processing import visualize_boxes

img_dir = 'data/VIB_page1'
config_file = 'form/template_VIB_page1.txt'
img_path = 'data/VIB_test_for_demo_06Mar/20200304 AICR Test-5.jpg'
img_path = 'data/VIB_page1/vib_page1-22.jpg'
#img_path = 'form/template_VIB/0001_ori.jpg'
img_path = ''
test_list = config.test_list
pretrained = config.pretrained_test
imgW = config.imgW
imgH = config.imgH
gpu = '0'  # config.gpu_test
#gpu = None
alphabet_path = config.alphabet_path
workers = 4
batch_size = 8

label = config.label
debug = False
if debug:
    batch_size = 1
alphabet = open(alphabet_path, encoding='UTF-8').read().rstrip()
nclass = len(alphabet) + 1
nc = 3
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def init_models(batch_sz):
    print('Init CRNN classifier')
    image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
    model = crnn.CRNN2(imgH, nc, nclass, 256)
    print('loading pretrained model from %s' % pretrained)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        model = model.cuda()
        image = image.cuda()
    else:
        print('Use CPU')

    converter = strLabelConverter(alphabet, ignore_case=False)

    image = Variable(image)
    model.eval()
    return model, converter, image


def init_post_processing(address_db_path, name_db, name_bigram, country_db):
    with open(address_db_path, 'rb') as handle:
        address_db = pickle.load(handle)
    name_db = load_name_corection(name_db, name_bigram)
    country_db = load_country_correction(csv_data=country_db)
    return address_db, name_db, country_db


def recognize(model, converter, image, list_obj, batch_sz, max_wh_ratio, max_iter=10000):
    numpy_list = []
    for obj in list_obj:
        numpy_list.append(obj.data)

    new_imgW = int(max_wh_ratio * imgH)
    # print('new_imgW',new_imgW)
    list_value = []
    val_dataset = NumpyListLoader(numpy_list)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sz,
        num_workers=workers,
        shuffle=False,
        collate_fn=alignCollate(new_imgW, imgH)
        # collate_fn=alignCollate(imgW, imgH)
    )
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    with torch.no_grad():
        for i in range(max_iter):
            data = val_iter.next()
            cpu_images, cpu_texts, _ = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            preds = model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            if (batch_sz > 1):
                list_value.extend(sim_pred)
            else:
                list_value.append(sim_pred)
            # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            # print('    ', raw_pred)
            # print(' =>', sim_pred)
            # if debug:
            #     print(' =>', sim_pred)
            #     inv_tensor = inv_normalize(cpu_images[0])
            #     cv_img = inv_tensor.permute(1, 2, 0).numpy()
            #     cv2.imshow('image data', cv_img)
            #     ch = cv2.waitKey(0)
            #     if ch == 27:
            #         break
    # assign again
    for idx, value in enumerate(list_value):
        setattr(list_obj[idx], 'value', value)
        if list_obj[idx].type in ['name', 'city', 'ward', 'district', 'street']:
            setattr(list_obj[idx], 'value_nlp', value)


def predict(list_img_path, batch_size=16, post_processing=True,
            address_db_path='symspellpy/db.pickle',
            address_csv_path='symspellpy/dvhcvn.csv',
            country_db_path='symspellpy/country-list.txt',
            name_dict="symspellpy/freq_name_dic.txt",
            name_bigram="symspellpy/freq_name_bigram.txt"):
    list_img_path = sorted(list_img_path)
    num_imgs = len(list_img_path)
    print('Predict', num_imgs, 'images')
    begin_init = time.time()
    model, converter, image = init_models(batch_size)
    address_db, name_db, country_db = init_post_processing(address_db_path, name_dict, name_bigram, country_db_path)
    end_init = time.time()
    print('Init time:', end_init - begin_init, 'seconds')

    transform_img_list = []
    for img in list_img_path:
        img_data = cv2.imread(img)
        img_resize = cv2.resize(img_data, (2480, 3508))
        trans_img = calib_image(img_resize)
        transform_img_list.append(trans_img)

    end_transform = time.time()
    print('Get data time:', end_transform - end_init, 'seconds')
    list_clImageInfor_demo, max_wh = extract_for_demo(transform_img_list, path_config_file=config_file, eraseline=False,
                                                      subtract_bgr=True, gen_bgr=False)

    num_samples = len(list_clImageInfor_demo)
    end_extract = time.time()
    print('Extract data time:', end_extract - end_transform, 'seconds')

    print('\nStart recognize')
    print('Number of samples', num_samples)
    recognize(model, converter, image, list_clImageInfor_demo, batch_size, max_wh)
    end_predict = time.time()
    processing_time = end_predict - end_extract
    print('Recognize time:', round(processing_time, 4), 'seconds. Speed:', round(num_samples / processing_time, 2),
          'fps')

    if post_processing:
        for idx, clImageInfor in enumerate(list_clImageInfor_demo):
            if clImageInfor.type == 'name':
                clImageInfor.value_nlp = correct_name(name_db, clImageInfor.value)[0]
            if clImageInfor.type == 'city':
                fixed_address = correct_address(db=address_db, csv_file=address_csv_path,
                                                street=list_clImageInfor_demo[idx - 3].value,
                                                ward=list_clImageInfor_demo[idx - 2].value,
                                                district=list_clImageInfor_demo[idx - 1].value, city=clImageInfor.value,
                                                max_edit_distance=4)
                list_clImageInfor_demo[idx-3].value_nlp = fixed_address['street_fixed']
                list_clImageInfor_demo[idx-2].value_nlp=fixed_address['ward_fixed']
                list_clImageInfor_demo[idx-1].value_nlp=fixed_address['district_fixed']
                clImageInfor.value_nlp=fixed_address['city_fixed']
            if clImageInfor.type=='country':
                clImageInfor.value_nlp = correct_country(clImageInfor.value, country_db)

    end_spell_checking = time.time()
    print('Spell checking time', end_spell_checking - end_predict, 'seconds')

    list_outputs = []
    num_fields = num_samples / num_imgs
    for idx, img in enumerate(list_img_path):
        print('\nResult of:', img)
        outputs = dict()
        start_idx = int(num_fields * idx)
        end_idx = int(num_fields * (idx + 1))
        for i in range(start_idx, end_idx):
            additional = ''
            if list_clImageInfor_demo[i].type in ['name', 'city', 'ward', 'district', 'street']:
                additional = ', fixed:'
            if post_processing:
                outputs[list_clImageInfor_demo[i].prefix]=list_clImageInfor_demo[i].value_nlp
            else:
                outputs[list_clImageInfor_demo[i].prefix] = list_clImageInfor_demo[i].value
            print(i -start_idx +1, list_clImageInfor_demo[i].prefix, ':', list_clImageInfor_demo[i].value, additional,
                  list_clImageInfor_demo[i].value_nlp)
        list_outputs.append(outputs)
        if debug:
            result = visualize_boxes(config_file, transform_img_list[idx])
            cv2.imshow('result', result)
            ch = cv2.waitKey(0)
            if ch == 27:
                break

    total_time = end_spell_checking - end_init
    print('\nTotal core engine time:', total_time, ', speed:', round(total_time / num_imgs, 2), 'seconds per image')
    return list_outputs


if __name__ == "__main__":
    imgs = get_list_file_in_folder(img_dir)
    for idx in range(len(imgs)):
        imgs[idx] = os.path.join(img_dir, imgs[idx])

    if img_path == '':
        predict(imgs, batch_size=batch_size)
    else:
        predict([img_path], batch_size=batch_size)
