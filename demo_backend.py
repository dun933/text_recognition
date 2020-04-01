import pathlib
import base64
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + '/classifier_CRNN')

import torch
from torch.autograd import Variable
from classifier_CRNN.models.utils import strLabelConverter
from classifier_CRNN.models import crnn, utils
import time
import os
import pickle, json

import cv2
from classifier_CRNN.utils.loader import alignCollate, NumpyListLoader

from torchvision import transforms
from classifier_CRNN.pre_processing.image_preprocessing import extract_for_demo, extract_for_demo_json, \
    clImageInfor_demo
# from classifier_CRNN.pre_processing.image_calibration import calib_image
from classifier_CRNN.symspellpy.address_spell_check import correct_address, correct_PlaceOfIssueId
from classifier_CRNN.symspellpy.general_spell_check import load_name_corection, correct_name, load_cpn_corection, \
    correct_cpn
from classifier_CRNN.symspellpy.general_spell_check import load_country_correction, correct_country
from classifier_CRNN.symspellpy.general_spell_check import load_relationship_correction, correct_relationship
from classifier_CRNN.symspellpy.general_spell_check import correct_date
from classifier_CRNN.form.form_processing import visualize_boxes, visualize_boxes_json
from classifier_CRNN.pre_processing.augment_functions import cnd_aug_resizePadding
from api_server.util import aicr_db

im = 5
img_list = ['I-1.png', 'II-2.png', 'III-5.png', 'IV-1.png', 'VI-1.jpg', 'VII-1.jpg']
img_dir = 'data/new_vbi_page1'
template_idx = 6
base_name = img_list[template_idx - 3]
config_file = 'data/SDV_invoices/' + base_name.replace('.png', '.txt').replace('.jpg', '.txt')
img_path = 'data/SDV_invoices/' + base_name
# img_path = 'classifier_CRNN/form/template_VIB/0001_ori.jpg'

pretrained = 'classifier_CRNN/ckpt/AICR_SDV_30Mar_No_update_hw_300_loss_1.25_cer_0.0076.pth'
imgW = 1600
imgH = 64

gpu = '1'
# gpu = None
alphabet_path = 'classifier_CRNN/data/char_246'
workers = 4
batch_size = 8

label = False
debug = False
calib = False
subtract_bgr = False
alphabet = open(alphabet_path, encoding='UTF-8').read().rstrip()
nclass = len(alphabet) + 1
nc = 3
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
fill_color = (255, 255, 255)  # (209, 200, 193)


def get_list_file_in_folder(dir, ext='png'):
    included_extensions = ['png', 'jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def init_models(batch_sz):
    print('Init CRNN classifier')
    image = torch.FloatTensor(batch_sz, 3, imgH, imgH)
    if imgH == 64:
        model = crnn.CRNN64(imgH, nc, nclass, 256)
    else:  # 32
        model = crnn.CRNN32(imgH, nc, nclass, 256)

    print('loading pretrained model from %s' % pretrained)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    if gpu is not None:
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


def init_post_processing(address_db_path, name_db, name_bigram, country_db, relationship_db):
    with open(address_db_path, 'rb') as handle:
        address_db = pickle.load(handle)
    name_db = load_name_corection(name_db, name_bigram)
    country_db = load_country_correction(csv_data=country_db)
    relationship_db = load_relationship_correction(csv_data=relationship_db)
    return address_db, name_db, country_db, relationship_db


def recognize(model, converter, image, list_obj, batch_sz, max_wh_ratio, max_iter=10000):
    numpy_list = []
    for obj in list_obj:
        numpy_list.append(obj.data)

    new_imgW = int(max_wh_ratio * imgH)
    transform_test = transforms.Compose([cnd_aug_resizePadding(new_imgW, imgH, fill=fill_color, train=False),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])
    list_value = []
    val_dataset = NumpyListLoader(numpy_list, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sz,
        num_workers=workers,
        shuffle=False
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
            preds_size = Variable(torch.IntTensor(
                [preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            if (batch_sz > 1):
                list_value.extend(sim_pred)
            else:
                list_value.append(sim_pred)

            if debug:
                # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
                # print('    ', raw_pred)
                print(' =>', sim_pred)
                inv_tensor = inv_normalize(cpu_images[0])
                cv_img = inv_tensor.permute(1, 2, 0).numpy()
                cv2.imshow('image data', cv_img)
                ch = cv2.waitKey(0)
                if ch == 27:
                    break
    # assign again
    for idx, value in enumerate(list_value):
        setattr(list_obj[idx], 'value', value)
        if list_obj[idx].type in ['name', 'city', 'ward', 'district', 'street', 'country'] or \
                list_obj[idx].id in [10, 39]:
            setattr(list_obj[idx], 'value_nlp', value)


init = False
if not init:
    begin_init = time.time()
    model, converter, image = init_models(batch_size)
    address_db, name_db, country_db, relationship_db = init_post_processing(
        'classifier_CRNN/symspellpy/data/db.pickle',
        'classifier_CRNN/symspellpy/data/freq_name_dic.txt',
        'classifier_CRNN/symspellpy/data/freq_name_bigram.txt',
        'classifier_CRNN/symspellpy/data/country-list.txt',
        'classifier_CRNN/symspellpy/data/relationship_list.txt'
    )
    company_list = 'classifier_CRNN/symspellpy/data/companies-list.txt'
    company_spell = load_cpn_corection(company_list)
    company_addr_list = 'classifier_CRNN/symspellpy/data/company-address-list.txt'
    company_addr_spell = load_cpn_corection(company_addr_list)
    money_word_list = 'classifier_CRNN/symspellpy/data/num_by_word.txt'
    money_word_spell = load_cpn_corection(money_word_list)
    end_init = time.time()
    print('Init time:', end_init - begin_init, 'seconds')
    init = True


def NLP_VIB_form(list_clImageInfor_final, start_idx_img, end_idx_img, address_csv_path):
    for i in range(start_idx_img, end_idx_img):
        if list_clImageInfor_final[i].type == 'name':
            list_clImageInfor_final[i].value_nlp = correct_name(
                name_db, list_clImageInfor_final[i].value)[0]
        if list_clImageInfor_final[i].type == 'city':
            fixed_address = correct_address(db=address_db, csv_file=address_csv_path,
                                            street=list_clImageInfor_final[i - 3].value,
                                            ward=list_clImageInfor_final[i - 2].value,
                                            district=list_clImageInfor_final[i - 1].value,
                                            city=list_clImageInfor_final[i].value,
                                            max_edit_distance=4)
            list_clImageInfor_final[i -
                                    3].value_nlp = fixed_address['street_fixed']
            list_clImageInfor_final[i -
                                    2].value_nlp = fixed_address['ward_fixed']
            list_clImageInfor_final[i -
                                    1].value_nlp = fixed_address['district_fixed']
            list_clImageInfor_final[i].value_nlp = fixed_address['city_fixed']
        if list_clImageInfor_final[i].type == 'country':
            list_clImageInfor_final[i].value_nlp = correct_country(
                list_clImageInfor_final[i].value, country_db).replace('\n', '')
        if list_clImageInfor_final[i].id == 10:
            list_clImageInfor_final[i].value_nlp = correct_PlaceOfIssueId(
                db=address_db, csv_file=address_csv_path, inp=list_clImageInfor_final[i].value)
        if list_clImageInfor_final[i].id == 39:
            list_clImageInfor_final[i].value_nlp = correct_relationship(
                list_clImageInfor_final[i].value.replace(' ', '_'), relationship_db).replace('\n', '')
        if list_clImageInfor_final[i].id == 2 or list_clImageInfor_final[i].id == 7:
            day = list_clImageInfor_final[i].value
            month = list_clImageInfor_final[i + 1].value
            year = list_clImageInfor_final[i + 2].value
            list_clImageInfor_final[i].value_nlp, list_clImageInfor_final[i + 1].value_nlp, list_clImageInfor_final[
                i + 2].value_nlp = correct_date(day, month, year)


def NLP_SDV_invoice(list_clImageInfor_final, start_idx_img, end_idx_img):
    for i in range(start_idx_img, end_idx_img):
        # print(i, list_clImageInfor_final[i].label)
        if list_clImageInfor_final[i].label == 'Tên bên bán hàng' or list_clImageInfor_final[i].label == 'Tên đơn vị':
            # print(list_clImageInfor_final[i].value)
            list_clImageInfor_final[i].value_nlp = correct_cpn(
                list_clImageInfor_final[i].value, company_spell, GT_file=company_list)
            # print(list_clImageInfor_final[i].value_nlp)
        if list_clImageInfor_final[i].label == 'Địa chỉ':
            list_clImageInfor_final[i].value_nlp = correct_cpn(
                list_clImageInfor_final[i].value, company_addr_spell)
        if list_clImageInfor_final[i].label == 'Số tiền viết bằng chữ':
            list_clImageInfor_final[i].value_nlp = correct_cpn(
                list_clImageInfor_final[i].value, money_word_spell)
        if list_clImageInfor_final[i].label == 'Ngày xuất':
            day = list_clImageInfor_final[i].value
            month = list_clImageInfor_final[i + 1].value
            year = list_clImageInfor_final[i + 2].value
            list_clImageInfor_final[i].value_nlp, list_clImageInfor_final[i + 1].value_nlp, list_clImageInfor_final[
                i + 2].value_nlp = correct_date(day, month, year)
    return list_clImageInfor_final


def predict_json(list_img_path, batch_size=16, post_processing=True,
                 address_csv_path='classifier_CRNN/symspellpy/data/dvhcvn.csv',
                 template_id=1, local=False, image_directory=None):
    list_img_path = sorted(list_img_path)
    num_imgs = len(list_img_path)
    print('Predict', num_imgs, 'images')

    begin_transform = time.time()
    transform_img_list = []
    for img in list_img_path:
        ext = 'jpg' if img.endswith('.jpg') else 'png'
        img_data = cv2.imread(img)
        if calib:
            # img_resize = cv2.resize(img_data, (2480, 3508))
            # trans_img = calib_image(img_resize)
            trans_img = img_data
            print('chua co hang dau')
        else:
            trans_img = img_data
        transform_img_list.append((trans_img, ext))

    end_transform = time.time()
    print('Get data time:', end_transform - begin_transform, 'seconds')

    list_clImageInfor_final = []
    list_mark_final = []

    if local:
        temp_json = aicr_db.get_template_config_local(template_id)
    else:
        temp_json = aicr_db.get_template_config(template_id)

    for (img, ext) in transform_img_list:
        subtract_bgr = False
        if template_id == 6:
            print('hard code to subtract background for tax in template 4')
            subtract_bgr = True
        list_clImageInfor_demo, max_wh, list_group_mark = extract_for_demo_json(img, temp_json=temp_json,
                                                                                eraseline=False,
                                                                                subtract_bgr=subtract_bgr,
                                                                                gen_bgr=False,
                                                                                image_directory = image_directory,
                                                                                image_ext = ext)

        # list_clImageInfor_demo, max_wh = extract_for_demo_([img], path_config_file=config_file,
        #                                                                         eraseline=False, subtract_bgr=False,
        #                                                                         gen_bgr=False)
        list_clImageInfor_final.extend(list_clImageInfor_demo)
        # list_mark_final.extend(list_group_mark)

    num_samples_img = len(list_clImageInfor_final)
    num_samples_mark = len(list_mark_final)
    end_extract = time.time()
    print('Extract data time:', end_extract - end_transform, 'seconds')

    print('\nStart recognize')
    print('Number of samples', num_samples_img)
    recognize(model, converter, image, list_clImageInfor_final, batch_size, max_wh)
    end_predict = time.time()
    processing_time = end_predict - end_extract
    print('Recognize time:', round(processing_time, 4), 'seconds. Speed:', round(num_samples_img / processing_time, 2),
          'samples/s')

    num_fields_img = num_samples_img / num_imgs
    num_fields_mark = num_samples_mark / num_imgs
    if post_processing:
        for idx, img in enumerate(list_img_path):
            start_idx_img = int(num_fields_img * idx)
            end_idx_img = int(num_fields_img * (idx + 1))
            list_clImageInfor_final = NLP_SDV_invoice(list_clImageInfor_final, start_idx_img, end_idx_img)

    end_spell_checking = time.time()
    print('Spell checking time', end_spell_checking - end_predict, 'seconds')

    list_outputs = []
    groups = temp_json["groups"]
    ignore_list = set()
    for g in groups:
        if g["hide_original_sources"]:
            ignore_list.update(g["sources"])

    field_value_set = dict()
    field_set = dict()
    for idx, img in enumerate(list_img_path):
        print('\nResult of:', img)
        outputs = []
        start_idx_img = int(num_fields_img * idx)
        end_idx_img = int(num_fields_img * (idx + 1))
        for i in range(start_idx_img, end_idx_img):
            additional = ''
            if list_clImageInfor_final[i].value_nlp != '':
                additional = ', fixed:'
                _value = list_clImageInfor_final[i].value_nlp
            else:
                _value = list_clImageInfor_final[i].value

            if list_clImageInfor_final[i].data_type == 'image' and _value is not None:
                with open(_value, "rb") as image_file:
                    _value = base64.b64encode(image_file.read()).decode('utf-8')

            field_value_set[list_clImageInfor_final[i].id] = _value
            field_set[list_clImageInfor_final[i].id] = list_clImageInfor_final[i]
            if list_clImageInfor_final[i].id not in ignore_list:
                outputs.append(build_field(list_clImageInfor_final[i].id,
                                           list_clImageInfor_final[i].prefix,
                                           _value,
                                           list_clImageInfor_final[i].data_type))

            print(list_clImageInfor_final[i].id, list_clImageInfor_final[i].label, ':',
                  list_clImageInfor_final[i].value,
                  additional, list_clImageInfor_final[i].value_nlp)
        start_idx_mark = int(num_fields_mark * idx)
        end_idx_mark = int(num_fields_mark * (idx + 1))

        for i in range(start_idx_mark, end_idx_mark):
            values = []
            print(list_mark_final[i].id, list_mark_final[i].label)
            for j, mark_info in enumerate(list_mark_final[i].list_mark_info):
                values.append(build_field(mark_info.id,
                                          mark_info.name,
                                          mark_info.ischecked))
                if mark_info.ischecked:
                    print('   ', mark_info.id, mark_info.label)
            field_value_set[list_mark_final[i].id] = values
            field_set[list_mark_final[i].id] = list_mark_final[i]
            if list_mark_final[i].id not in ignore_list:
                outputs.append(build_field(list_mark_final[i].id,
                                           list_mark_final[i].name,
                                           values,
                                           list_mark_final[i].data_type))

        for g in groups:
            if len(g["sources"]) == 0:
                continue

            if g["group_type"] == "split":
                s_values = []
                for i, s in enumerate(g["sources"]):
                    field = field_set[s]
                    field_value = field_value_set[s]
                    if isinstance(field, clImageInfor_demo):
                        field_name = field.prefix
                    else:
                        field_name = field.name
                    data_type = "text" if field.data_type is None else field.data_type
                    s_values.append(build_field(field.id, field_name, field_value, data_type))

                outputs.append(build_field(10000 + g["id"], g["name"], s_values, "group"))
            elif g["group_type"] == "combine":
                c_values = ""
                field_value_separator = g["data_separator"]
                for index, s in enumerate(g["sources"]):
                    separator = "" if index == 0 else field_value_separator
                    if field_value_set[s] != '':
                        c_values = "{0}{1}{2}".format(c_values, separator, field_value_set[s]).strip()

                if g["hide_original_sources"]:
                    g_id = field_set[g["sources"][0]].id
                else:
                    g_id = 10000 + g["id"]
                outputs.append(build_field(g_id, g["name"], c_values, "text"))
        outputs.sort(key=lambda e: e["id"])
        list_outputs.append(outputs)
        if debug:
            result = visualize_boxes_json(temp_json, transform_img_list[idx])
            cv2.imshow('result', result)
            ch = cv2.waitKey(0)
            if ch == 27:
                break

    total_time = end_spell_checking - begin_transform
    print('\nTotal core engine time:', total_time, ', speed:',
          round(total_time / num_imgs, 2), 'seconds per image')
    return list_outputs if len(list_outputs) > 1 else list_outputs[0]


def build_field(id, name, value, data_type=None):
    field = dict()
    field["id"] = id
    field["name"] = name
    if data_type is not None:
        field["data_type"] = data_type
    field["value"] = value

    return field


if __name__ == "__main__":
    debug = False
    batch_size = 1
    imgs = get_list_file_in_folder(img_dir)
    for idx in range(len(imgs)):
        imgs[idx] = os.path.join(img_dir, imgs[idx])

    if img_path == '':
        predict_json(imgs, template_id=template_idx, batch_size=batch_size, local=True)
    else:
        predict_json([img_path], template_id=template_idx, batch_size=batch_size, local=True)
