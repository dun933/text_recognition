import torch
from torch.autograd import Variable
from models.utils import strLabelConverter
import models.crnn as crnn
import time, os

import cv2
from loader import alignCollate, NumpyListLoader
import models.utils as utils
import config
from torchvision import transforms
from image_preprocessing import extract_for_demo
from image_calibration import calib_image
from symspellpy2.address_spell_check import load_address_correction, correct_address
from symspellpy2.name_spell_check import load_name_corection, correct_name

img_dir = '/data/data_imageVIB/vib_page1'
test_list=config.test_list
pretrained = config.pretrained_test
imgW = config.imgW
imgH = config.imgH
gpu = config.gpu_test
alphabet_path = config.alphabet_path
workers = 4
batch_size = 8

label = config.label
debug = True
if debug:
    batch_size = 1
alphabet = open(alphabet_path).read().rstrip()
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
        print('Use GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        model = model.cuda()
        image = image.cuda()

    converter = strLabelConverter(alphabet, ignore_case=False)


    image = Variable(image)
    model.eval()
    return model, converter, image

def init_post_processing(address_db_path, name_db, name_bigram):
    address_db = load_address_correction(address_db_path)
    name_db = load_name_corection(name_db, name_bigram)
    return name_db, address_db

def recognize(model, converter, image, numpy_list, batch_sz, max_iter = 10000):
    list_value=[]
    val_dataset = NumpyListLoader(numpy_list)
    num_files = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sz,
        num_workers=workers,
        shuffle=False,
        collate_fn=alignCollate(imgW, imgH)
    )
    print('Start predict')
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    print('Number of samples', num_files)
    begin = time.time()
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
            if (batch_sz>1):
                list_value.extend(sim_pred)
            else:
                list_value.append(sim_pred)
            #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            #print('    ', raw_pred)
            #print(' =>', sim_pred)
            #print(sim_pred)
            if debug:
                print(' =>', sim_pred)
                inv_tensor = inv_normalize(cpu_images[0])
                cv_img = inv_tensor.permute(1, 2, 0).numpy()
                cv2.imshow('image data', cv_img)
                cv2.waitKey(0)
    end = time.time()
    processing_time = end - begin
    #print('Processing time:',processing_time)
    #print('Speed:', num_files / processing_time, 'fps')
    return list_value


def predict(list_img_path, batch_size = 16, post_processing=True,
            config_file= 'form/template_VIB_page1_demo.txt',
            address_db='symspellpy2/full_db.pickle',
            name_dict= "symspellpy2/freq_name_dic.txt",
            name_bigram= "symspellpy2/freq_name_bigram.txt"):
    if(len(list_img_path)>1):
        print('Please use 1 image only')
    print('predict image:',list_img_path[0])
    begin_init = time.time()
    model, converter, image = init_models(batch_size)
    name_db, address_db = init_post_processing(address_db, name_dict, name_bigram)
    end_init = time.time()
    print('Init time', end_init-begin_init,'seconds')

    for img in list_img_path:
        img_data = cv2.imread(img)
        img_resize = cv2.resize(img_data, (2480, 3508))
        trans_img = calib_image(img_resize)
        img_list.append(trans_img)

    end_transform = time.time()
    print('Get data time', end_transform - end_init, 'seconds')
    list_obj = extract_for_demo(img_list, path_config_file=config_file, eraseline=False)

    pred_output=dict()
    numpy_data = []
    for obj in list_obj:
        pred_output[obj.prefix]=''
        numpy_data.append(obj.data)

    end_extract = time.time()
    print('Extract data time', end_extract - end_transform, 'seconds')
    list_value = recognize(model, converter, image, numpy_data, batch_size)
    print(list_value)

    print ('recognized name:',list_value[0])
    print ('recognized street:',list_value[4], 'ward:',list_value[5],'district:',list_value[6],'city:',list_value[7])

    fixed_address = correct_address(db=address_db, street=list_value[4], ward=list_value[5], district=list_value[6], city=list_value[7])
    fixed_name = correct_name(name_db, list_value[0])

    print ('fixed name:',fixed_name)
    print ('fixed address:',fixed_address)

    end_predict = time.time()
    print('Recognize time', end_predict - end_extract, 'seconds')

if __name__== "__main__":
    img_list = []
    imgs = get_list_file_in_folder(img_dir)
    for idx in range(len(imgs)):
        imgs[idx]=os.path.join(img_dir,imgs[idx])

    img_path='/data/data_imageVIB/vib_page1/vib_page1-15.jpg'
    predict([img_path], batch_size=batch_size)

