import os, glob, time, cv2, sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from datetime import datetime
from config.config_manager import ConfigManager
import shutil
from matplotlib import rc, pyplot as plt
import itertools, codecs
from predict_classifier2 import visual_error_predict, get_confusion_matrix, plot_confusion_matrix

configfile = 'config/classifier_config.ini'
configmanager = ConfigManager(configfile)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier1_infer)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
save_dir = configmanager.save_classifier1_dir + '/predict_classifier1_' + training_time
input_size = configmanager.input_size_classifier1[0]

test_dir = configmanager.test_classifier1_dir
model_file = configmanager.weight_path_classifier1
class_name = ['A', 'E', 'I', 'O', 'U', 'Y', 'alp', 'num', 'sym']

class_order = {}
for i in range(len(class_name)):
    class_order[class_name[i]] = i

def get_list_file_in_folder(dir, ext=None):
    if ext is None:
        ext = ['jpg', '.png']
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def load_image(image_file):
    test_img = image.load_img(image_file, color_mode='grayscale', target_size=(input_size, input_size))
    test_img = image.img_to_array(test_img)
    test_img = test_img * (1. / 255.)
    x = test_img[np.newaxis, :]
    return x

def predict(model_path):
    result='Begin predict with model: '+ str(model_file)+'\n'
    print(result)
    model = load_model(model_path)
    # model.load_weights('outputs/train_2019-10-03_16-37_9319/weights_05-0.22.hdf5')
    true_pred = {}
    sample = {}
    total_true_pred = 0
    total_sample = 0
    for cls in class_name:
        # print('Test class: ',cls)
        img_list = get_list_file_in_folder(os.path.join(test_dir, cls))
        true_pred[cls] = 0
        sample[cls] = 0
        for img in img_list:
            # print(img)
            data = load_image(os.path.join(test_dir, cls, img))
            ret_classes = model.predict(data)
            max_idx = np.argmax(ret_classes, axis=1)
            max_val = np.amax(ret_classes)
            # print('class name:', class_name[max_idx[0]], ', conf:', max_val)
            sample[cls] += 1
            with codecs.open(result_file, 'a', encoding='utf-8') as fr1:
                fr1.write(class_name[max_idx[0]] + ' ' + cls + ' ' + os.path.join(test_dir, cls, img) + '\n')
            if class_name[max_idx[0]] == cls:
                true_pred[cls] += 1
            else:
                with codecs.open(error_file, 'a', encoding='utf-8') as fh1:
                    fh1.write(class_name[max_idx[0]] + ' ' + cls + ' ' + os.path.join(test_dir, cls, img) + '\n')
        total_true_pred += true_pred[cls]
        total_sample += sample[cls]
        accuracy = float(true_pred[cls]) / float(sample[cls])
        line ='Class ' + str(cls)+ ', True pred: '+ str(true_pred[cls])+ ', Sample: '+ str(sample[cls])+ ', Acc:'+ str(accuracy)
        print(line)
        result+=line+'\n'

    accuracy = float(total_true_pred) / float(total_sample)
    final_line='All class: True pred: ' + str(total_true_pred)+', Sample: ' + str(total_sample)+ ', Acc: ' + str(accuracy)
    print(final_line)
    result+=final_line
    with open(os.path.join(save_dir, 'predict.log'), "w") as f:
        f.write(result)

# input: a directory with png images
# output: a directory with png images which are result of classifier1 belong to 9 classes
def predict_from_dir(model_path, input_dir, output_dir):
    print('Begin predict with model:', model_file)
    for cls in class_name:
        if not os.path.exists(os.path.join(output_dir, cls)):
            os.makedirs(os.path.join(output_dir, cls))
    model = load_model(model_path)
    img_list = get_list_file_in_folder(input_dir)
    for idx, img in enumerate(img_list):
        print(img)
        data = load_image(os.path.join(input_dir, img))
        ret_classes = model.predict(data)
        max_idx = np.argmax(ret_classes, axis=1)
        max_val = np.amax(ret_classes)
        print('predict class:', class_name[max_idx[0]], ', conf:', max_val)
        src_file = os.path.join(input_dir, img)
        dst_file = os.path.join(output_dir, class_name[max_idx[0]], img)
        shutil.copy(src_file, dst_file)
    print('Finish!')

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    error_file = os.path.join(save_dir, "error.log")
    result_file = os.path.join(save_dir, "result.log")

    codecs.open(error_file, 'w', encoding='utf-8').close()
    codecs.open(result_file, 'w', encoding='utf-8').close()

    predict(model_file)
    print('Done')
    # visualize error cases
    visual_error_predict(save_dir, error_file)

    # visualize confusion matrix
    cnf_matrix = get_confusion_matrix(result_file, class_order)
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    plot_confusion_matrix(cnf_matrix, classes=class_name,
                          title='Confusion matrix')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    #plt.show()
