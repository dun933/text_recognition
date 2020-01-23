import os, glob, time, cv2, sys
import warnings, codecs

warnings.filterwarnings("ignore")
import numpy as np
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from datetime import datetime
from config.config_manager_chinese import ConfigManager
from model.model_builder import InceptionResNetV2
from matplotlib import rc, pyplot as plt
import itertools
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Dense, GlobalMaxPooling2D, Input
from keras.models import model_from_json
import json
configfile = 'config/classifier_chinese_config.ini'
configmanager = ConfigManager(configfile)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier_infer)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

input_size_w = configmanager.input_size_classifier[0]
input_size_h = configmanager.input_size_classifier[1]
chanel_size = configmanager.input_size_classifier[2]

save_dir = configmanager.save_classifier_dir + '/predict_classifier_chinese_' + training_time

test_dir = configmanager.test_classifier_dir
weight_file = configmanager.weight_path_classifier
model_file = configmanager.model_path_classifier

class_name_vs_id = json.loads(open(configmanager.class_map, 'r').read())
nb_classes = len(class_name_vs_id)
class_name = [x for x in class_name_vs_id]
# print(nb_classes)
# exit()
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

def showImagesMatrix(save_dir, images, col=20):
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)
    number_of_files = len(images)
    row = number_of_files / col
    if number_of_files % col != 0:
        row += 1
    for n, (title, image) in enumerate(images):
        a = fig.add_subplot(row, col, n + 1)
        a.title.set_text(title + ' ')
        plt.imshow(image, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "error.png"))


def visual_error_predict(save_dir, log_file):
    list_error = []
    with codecs.open(log_file, "r", encoding='utf8') as fh1:
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            img = cv2.imread(splitLine[2], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
            line = [splitLine[0] + ' - ' + splitLine[1], img]
            list_error.append(line)

    showImagesMatrix(save_dir, list_error)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_confusion_matrix(log_file, class_order):
    def confusion_matrix(y_true, y_pred, class_order):
        N = len(class_order)
        cm = np.zeros((N, N), dtype=int)
        for n in range(y_true.shape[0]):
            cm[class_order[y_true[n]], class_order[y_pred[n]]] += 1
        return cm

    list_result = []
    with codecs.open(log_file, "r", encoding='utf8') as fh1:
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            line = [splitLine[0], splitLine[1]]
            list_result.append(line)
    result_arr = np.asarray(list_result)
    y_pred, y_true = result_arr.T
    return confusion_matrix(y_true, y_pred, class_order)

def load_image(image_file):
    origimg = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(origimg, (input_size_w, input_size_h))
    img = img * (1. / 255.)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=2)
    x = img[np.newaxis, :]
    return x


def load_model_from_json(json_model_path):
    json_file = open(json_model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


def convert_multigpu_to_one(json_model_path, weight_path):
    model_ = load_model_from_json(json_model_path)
    model = multi_gpu_model(model_, gpus=2)
    model.load_weights(weight_path)

    return model.layers[-2]  # get single GPU model weights

def predict(model_path, weight_path):
    print('Begin predict with model:', weight_path)

    model_ = convert_multigpu_to_one(model_path, weight_path)
    list_gpus = configmanager.gpu_num_classifier_infer.split(',')

    # model = multi_gpu_model(model_, gpus=len(list_gpus))
    # model = load_model(model_path)
    model = model_
    # model.load_weights(weight_path)
    true_pred = {}
    sample = {}
    total_true_pred = 0
    total_sample = 0
    for cls in class_name:
        img_list = get_list_file_in_folder(os.path.join(test_dir, cls))
        true_pred[cls] = 0
        sample[cls] = 0
        for img in img_list:
            data = load_image(os.path.join(test_dir, cls, img))
            ret_classes = model.predict(data)
            max_idx = np.argmax(ret_classes, axis=1)
            max_val = np.amax(ret_classes)

            sample[cls] += 1
            with codecs.open(result_file, 'a', encoding='utf-8') as fr1:
                fr1.write(class_name[max_idx[0]] + ' ' + cls + ' ' + os.path.join(test_dir, cls, img) + '\n')
            if class_name_vs_id[cls] == max_idx[0]:
                true_pred[cls] += 1
            else:
                with codecs.open(error_file, 'a', encoding='utf-8') as fh1:
                    fh1.write(class_name[max_idx[0]] + ' ' + cls + ' ' + os.path.join(test_dir, cls, img) + '\n')

        total_true_pred += true_pred[cls]
        total_sample += sample[cls]
        if sample[cls] > 0:
            accuracy = float(true_pred[cls]) / float(sample[cls])
            print('Class: ', cls, ', True pred:', true_pred[cls], ', Sample:', sample[cls], ', Acc:', accuracy)
        else:
            print('Class: ', cls, ' has no sample in test set')

    accuracy = float(total_true_pred) / float(total_sample)
    print('All class: True pred:', total_true_pred, ', Sample:', total_sample, ', Acc:', accuracy)


if __name__ == "__main__":
    orig_stdout = sys.stdout
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, "predict.log")
    error_file = os.path.join(save_dir, "error.log")
    result_file = os.path.join(save_dir, "result.log")

    codecs.open(error_file, 'w', encoding='utf-8').close()

    codecs.open(result_file, 'w', encoding='utf-8').close()
    print('Please check output of predict process in:', log_file)
    with open(log_file, 'w') as f:
        sys.stdout = f
        predict(model_file, weight_file)
        print('Done')
    sys.stdout = orig_stdout

    # visualize error cases
    visual_error_predict(save_dir, error_file)

    # visualize confusion matrix
    cnf_matrix = get_confusion_matrix(result_file, class_order)
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    plot_confusion_matrix(cnf_matrix, classes=class_name,
                          title='Confusion matrix')
    plt.show()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
