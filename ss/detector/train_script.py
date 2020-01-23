"""
aicr box detector training script. DSSD model
"""
import os, json, argparse

import warnings
warnings.filterwarnings("ignore")
from config.config_manager import ConfigManager

import os, cv2, math, sys, time
import numpy as np
import pickle

from imageio import imread
import glob, random
from datetime import datetime

import keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model
from keras.preprocessing import image
from keras.utils import plot_model

from model.model_builder import SSD_AICR
#from model.model_builder_tuananh import SSD_AICR

# from ssd_training import MultiboxLoss
from loss.loss_builder import MultiboxLoss
from utils.ssd_utils import BBoxUtility

from utils.generator_multi import Generator
from utils.get_aicr_data_from_json_cls4 import Json_preprocessor
from utils.get_aicr_data_from_XML_cls4 import XML_preprocessor

from parallel_model import ParallelModel
# get date
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
np.set_printoptions(suppress=True)
from keras import backend as k
print("run")
configfile = 'config/vietnamese_config.ini'
configManager = ConfigManager(configfile)

save_dir = configManager.save_dir + '/train_' + training_time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configManager.gpu_num

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))

voc_classes = configManager.classes
NUM_CLASSES = len(voc_classes) + 1
input_size = 320
channels = 3
batch_size = 32
save_interval = 1
nb_epoch = 80
num_worker=16
base_lr = 0.001 #1

input_shape = (input_size, input_size, channels)
priors = pickle.load(open(configManager.prior_pkl_file, 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

class writer:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, text):
        for w in self.writers:
            w.write(text)
    def flush(self):
        pass

def schedule(epoch, decay=0.9):
    lr = 1e-3
    lr = lr / 2
    if epoch < 2:
        lr = lr
    elif epoch < 4:
        lr = lr / 2
    elif epoch < 8:
        lr = lr / 4
    elif epoch < 10:
        lr = lr / 8
    elif epoch < 12:
        lr = lr / 12
    elif epoch < 14:
        lr = lr / 16
    elif epoch < 16:
        lr = lr / 24
    elif epoch < 18:
        lr = lr / 30
    else:
        lr = np.random.randint(1, 5) * 1e-5

    print('==========[learning rate] : ' + str(lr) + '==========')
    return lr
# lrScheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.8
    epochs_drop = 2.0
    epoch_new = epoch % 10
    lrate = initial_lrate * math.pow(drop, epoch_new)
    print('==========[learning rate] : ' + str(lrate) + '==========')
    return lrate

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        train_time=time.time() - self.epoch_time_start
        print("Training time:",train_time,"seconds")
        self.times.append(train_time)

def generate_train_val():
    print(configManager.train_dir)
    train_annot_path = os.path.join(configManager.train_dir, 'annots/')
    train_image_path = os.path.join(configManager.train_dir, 'images')
    val_annot_path = os.path.join(configManager.val_dir, 'annots/')
    val_image_path = os.path.join(configManager.val_dir, 'images')

    train_gt = Json_preprocessor(train_annot_path, num_classes=len(voc_classes), remove_unused=True).data
    train_keys = sorted(train_gt.keys())
    random.shuffle(train_keys)
    val_gt = Json_preprocessor(val_annot_path, num_classes=len(voc_classes), remove_unused=True).data
    val_keys = sorted(val_gt.keys())

    print("train size:{0} , val size:{1}".format(len(train_keys), len(val_keys)))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    train_gen = Generator(train_gt, bbox_util, batch_size, train_image_path + '/', train_keys,
                          (input_shape[0], input_shape[1]), do_crop=False)
    val_gen = Generator(val_gt, bbox_util, batch_size, val_image_path + '/', val_keys, (input_shape[0], input_shape[1]),
                        do_crop=False)
    return train_keys, val_keys, train_gen, val_gen

train_keys, val_keys, train_gen, val_gen = generate_train_val()
model = SSD_AICR(input_shape, num_classes=NUM_CLASSES)

if configManager.restore_ckpt_flag:
    model.load_weights(configManager.restore_ckpt_path)
#model.summary()

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=10, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint(os.path.join(save_dir,'weights_{epoch:02d}-{val_loss:.2f}.hdf5'),
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=False, 
                             save_weights_only=True,
                             mode='min',
                             period=save_interval)
time_callback = TimeHistory()
reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tensorboard = TensorBoard(log_dir=os.path.expanduser(save_dir),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

#callbacks = [early_stop, checkpoint, reduce_lr, tensorboard, time_callback]
callbacks = [checkpoint, reduce_lr, tensorboard, time_callback]

optim = keras.optimizers.Adam(lr=base_lr, amsgrad=True)
#model = ParallelModel(model, 2)
model.compile(optimizer=optim, loss=MultiboxLoss(NUM_CLASSES, alpha=0.9, neg_pos_ratio=3.0).compute_loss)

train_steps = (int)(len(train_keys) / batch_size /2)
val_steps = (int)(len(val_keys) / batch_size /2)

if train_steps < 1:
    train_steps = 1
if val_steps < 1:
    val_steps = 1

saved = sys.stdout
log_file = os.path.join(save_dir, "train.log")
print('Please check output of training process in:', log_file)
f = open(log_file, 'w')
sys.stdout = writer(sys.stdout, f)

print('train step: {0}, val step: {1}'.format(train_steps, val_steps))
history = model.fit_generator(generator=train_gen, 
                              steps_per_epoch=train_steps,
                              epochs=nb_epoch, 
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              workers = num_worker,
                              max_queue_size = 32,
                              use_multiprocessing = True)

sys.stdout = saved
f.close()

