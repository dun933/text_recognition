import os, glob, time, cv2, sys
import warnings, codecs

warnings.filterwarnings("ignore")
import numpy as np
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from datetime import datetime
from config.config_manager import ConfigManager
from matplotlib import rc, pyplot as plt
import itertools

configfile = 'config/classifier_config.ini'
configmanager = ConfigManager(configfile)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier2_infer)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
sub_group = configmanager.group_classifier2
input_size_w = configmanager.input_size_classifier2[0]
input_size_h = configmanager.input_size_classifier2[1]
save_dir = configmanager.save_classifier2_dir + '/predict_classifier2_' + sub_group + '_' + training_time

test_dir = configmanager.test_classifier2_dir
model_file = configmanager.weight_path_classifier2
class_list={} #total 227-5470  -->201 after ignore lower char
class_list['A'] = configmanager.class_list_A # 37-444
class_list['E'] = configmanager.class_list_E #24-270
class_list['O'] = configmanager.class_list_O #45-836 --> 26 after ignore lower char
class_list['I'] = configmanager.class_list_I #30-970
class_list['U'] = configmanager.class_list_U #24-198
class_list['Y'] =  configmanager.class_list_Y  #16-195 --> 13 after ignore lower char
class_list['num'] = configmanager.class_list_num #14-849 --> 13 after ignore lower char
class_list['sym'] = configmanager.class_list_sym #20-255
class_list['alp'] = configmanager.class_list_alp #17-1453 --> 14 after ignore lower char
merge_list= configmanager.ignore_class_list
class_name = [x.replace('.','dot').replace('/','slash') for x in class_list[sub_group] if x not in merge_list]
#class_name=['A']

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

def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

def showImagesMatrix(save_dir, images, col=20):
    fig = plt.figure(figsize=(38.4, 21.6), dpi=100)
    number_of_files = len(images)
    row = number_of_files / col
    if number_of_files % col != 0:
        row += 1
    for n, (title, image) in enumerate(images):
        a = fig.add_subplot(row, col, n + 1)
        a.title.set_text(title + ' ')
        plt.imshow(frame_image(image,1), cmap='Greys_r')
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
            img = cv2.imread(splitLine[3], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
            save_name=(os.path.basename(splitLine[3]).split('.')[0])[-12:]
            line = [splitLine[2]+' '+ splitLine[0] + ' - GT ' + splitLine[1]+'\n'+save_name, img]
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
    test_img=image.load_img(image_file,color_mode='grayscale', target_size=(input_size_w, input_size_h))
    test_img=image.img_to_array(test_img)
    test_img = test_img * (1. / 255.)
    x = test_img[np.newaxis, :]
    return x

def predict(model_path):
    result='Begin predict with model: '+ str(model_file)+'\n'
    print(result)
    model = load_model(model_path)
    #print(model.summary())
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
            if class_name[max_idx[0]] == cls:
                true_pred[cls] += 1
            else:
                with codecs.open(error_file, 'a', encoding='utf-8') as fh1:
                    fh1.write(class_name[max_idx[0]] + ' ' + cls + ' ' +str(round(max_val,2))+' '+ os.path.join(test_dir, cls, img) + '\n')

        total_true_pred += true_pred[cls]
        total_sample += sample[cls]
        line ='Class ' + str(cls)+ ' has no sample in test set'
        if sample[cls] > 0:
            accuracy = float(true_pred[cls]) / float(sample[cls])
            line ='Class ' + str(cls)+ ', True pred: '+ str(true_pred[cls])+ ', Sample: '+ str(sample[cls])+ ', Acc:'+ str(accuracy)
        print(line)
        result+=line+'\n'
    accuracy = float(total_true_pred) / float(total_sample)
    final_line='All class: True pred: ' + str(total_true_pred)+', Sample: ' + str(total_sample)+ ', Acc: ' + str(accuracy)
    print(final_line)
    result += final_line
    with open(os.path.join(save_dir, 'predict.log'), "w") as f:
        f.write(result)

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
