import warnings
warnings.filterwarnings("ignore")
from keras.layers import Dense, GlobalMaxPooling2D, Input
from keras.engine import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import json, time, keras
import os, sys
from datetime import datetime
from config.config_manager_chinese import ConfigManager
from model.model_builder import InceptionResNetV2
from keras.utils.training_utils import multi_gpu_model
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from general_utils.lmdb_core import LMDB_CORE
from general_utils.lmdb_core import Mode_Chinese_Classify
import math


class_name_ = ''
mode_class  = ''
if len(sys.argv) == 3:
    class_name_ = sys.argv[1]
    mode_class  = sys.argv[2]
    print('class_name_ ',class_name_)
    print('mode_class ',mode_class)
else:
    print(len(sys.argv))
    print("argument input error")
    exit()


configfile = 'config/classifier_chinese_config.ini'
configmanager = ConfigManager(configfile)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
# sub_group = configmanager.group_classifier
save_dir = configmanager.save_classifier_dir+ '/train_classifier_chinese_' +class_name_+ training_time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier_train)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", (configmanager.gpu_num_classifier_train))

is_load_model = configmanager.restore_ckpt
loaded_model_path = configmanager.restore_ckpt_path


chinese_file_path = "../textimg_data_generator_dev/dataprovision/config/chinese.txt"
class_list_chinese = list()
if chinese_file_path is not None:
    with open(chinese_file_path) as fp:
        class_list_chinese = [c for c in fp.read(-1)]

classes_symbol = '*:,@$.-(#%\'\")/~!^&_+={}[]\;<>?※”'
class_list_symbol = [x for x in classes_symbol]


classes_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
class_list_alphabet = [x for x in classes_alphabet]

classes_number = '0123456789'
class_list_number = [x for x in classes_number]

input_size_w = configmanager.input_size_classifier[0]
input_size_h = configmanager.input_size_classifier[1]
chanel_size = configmanager.input_size_classifier[2]

save_interval = 1
base_lr = 3e-4
nb_epochs=2
num_thread=10

# data load
train_data_path = configmanager.train_classifier_dir
val_data_path = configmanager.val_classifier_dir

list_gpus = configmanager.gpu_num_classifier_train.split(',')
train_batch_size = configmanager.batch_size_classifier * len(list_gpus)

val_batch_size = configmanager.batch_size_classifier

# train data set
# train_datagen = ImageDataGenerator( rescale = 1. / 255)
# # width_shift_range=6,
# # rotation_range=6)
# train_generator = train_datagen.flow_from_directory(
#     train_data_path,
#     target_size=(input_size_h, input_size_w),
#     batch_size=train_batch_size,
#     color_mode='grayscale',
#     shuffle=True,
#     # seed=49,
#     class_mode='categorical')

# validation data set
# val_datagen = ImageDataGenerator(rescale=1. / 255)
# # width_shift_range=6,
# # rotation_range=6)
# val_generator = val_datagen.flow_from_directory(
#     val_data_path,
#     target_size=(input_size_h, input_size_w),
#     batch_size=val_batch_size,
#     color_mode='grayscale',
#     shuffle=False,
#     # seed=49,
#     class_mode='categorical')

# train_classmap = train_generator.class_indices
# val_classmap = val_generator.class_indices

# print("train_generator class index: {}".format(train_generator.class_indices))
# print("val_generator class index: {}".format(val_generator.class_indices))

path_train_dir = '/home/advlab/data_thang/valid_data_15_11_2019_13_53'
path_valid_dir = '/home/advlab/data_thang/valid_data_15_11_2019_13_53'
Mode_Class = None
if mode_class[0] == '0':
    Mode_Class = Mode_Chinese_Classify.FILTER_LAGNGUAGE
elif mode_class[0] == '1':
    Mode_Class = Mode_Chinese_Classify.CHINESE
elif mode_class[0] == '2':
    Mode_Class = Mode_Chinese_Classify.NUMBER
elif mode_class[0] == '3':
    Mode_Class = Mode_Chinese_Classify.SYMBOL
elif mode_class[0]== '4':
    Mode_Class = Mode_Chinese_Classify.ENGLISH
else:
    print("error Mode")
    exit()

train_gen = LMDB_CORE(path_train_dir, 'gray', input_size_h, input_size_w, train_batch_size, Mode_Class,26350)
nb_classes = train_gen.get_number_Class()
numb_samples_train = train_gen.get_number_Samples()
print('numb_samples_train ', numb_samples_train)
val_gen = LMDB_CORE(path_valid_dir, 'gray', input_size_h, input_size_w, val_batch_size, Mode_Class)
numb_samples_valid = val_gen.get_number_Samples()
print('numb_samples_valid ', numb_samples_valid)




def schedule(epoch, decay=0.9):
    lr = base_lr * 0.9 ** (epoch)
    print('##### learning rate of epoch ', int(epoch) + 1, ' : ', lr)
    train_gen.globals_count = 0
    return lr

# with open(os.path.join(save_dir, "classmap.json"), 'w') as fd:
#     json.dump(train_classmap, fd)

inputs = Input(shape=(input_size_h, input_size_w, chanel_size))

x = InceptionResNetV2(inputs)
x = GlobalMaxPooling2D()(x)



predictions = Dense(nb_classes, activation='softmax')(x)

model_ = Model(inputs=inputs, outputs=predictions)
# model_.save(os.path.join(save_dir, 'model.json'))

model = multi_gpu_model(model_, gpus=len(list_gpus))

early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=6,
                               mode='min',
                               verbose=2)

model_json = model_.to_json()
with open(configmanager.save_classifier_dir + '/model_single_gpu.' + str(training_time) + '.json', "w") as json_file:
    json_file.write(model_json)

if is_load_model:
    model.load_weights(loaded_model_path)

checkpoint = ModelCheckpoint(
        os.path.join(save_dir, 'weights_{epoch:02d}-{val_loss:.2f}.hdf5'),
        monitor='val_loss',
        verbose=2,
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        period=save_interval)

reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
cbs = [early_stop, checkpoint, reduce_lr]
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

#print(model.summary())
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
log_file = os.path.join(save_dir, "train.log")
print('Please check output of training process in:', log_file)
f = open(log_file, 'w')
#sys.stdout = f
# save the classmap in output dir
# with open(os.path.join(save_dir, "classmap.json"), 'w') as fd:
#     json.dump(train_classmap, fd)
with open(os.path.join(save_dir, "classmap.json"), 'w+') as fd:
    str_w = ''
    for k, val in train_gen.output_label_integer.items():
        str_w += str(k) + ' ' + str(val[0]) + '\n'
    fd.write(str_w)
    fd.close()
print('train_dir:', train_data_path)
print('val_dir:', val_data_path)
if num_thread > 1:
    is_use_multiprocessing = True
hist = model.fit_generator(
        train_gen.image_generator(),
        verbose=1,
        steps_per_epoch=train_gen.get_steps_per_epoch(),
        epochs=nb_epochs,
        use_multiprocessing=is_use_multiprocessing,
        validation_data=val_gen.image_generator(),
        validation_steps=val_gen.get_steps_per_epoch(),
        workers=num_thread,
        callbacks=cbs)
f.close()
