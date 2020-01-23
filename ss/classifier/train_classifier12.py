import warnings
warnings.filterwarnings("ignore")
from keras.layers import Dense, GlobalMaxPooling2D, Input
from keras.engine import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import json, time, keras
import os, sys
from datetime import datetime
from config.config_manager import ConfigManager
from model.model_builder import InceptionResNetV2
from keras.utils.training_utils import multi_gpu_model

configfile='config/classifier_config.ini'
configmanager = ConfigManager(configfile)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
#classifier12 with 201 classes
save_dir = configmanager.save_classifier12_dir+ '/train_classifier12_all_228classes_'+training_time
multi_gpus=False
gpu_num=configmanager.gpu_num_classifier12_train
list_gpus = gpu_num.split(',')
if(len(list_gpus)>1):
    print('training with multi gpu:',gpu_num)
    multi_gpus = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

input_size_w = configmanager.input_size_classifier12[0]
input_size_h = configmanager.input_size_classifier12[1]
pre_trained_weight=configmanager.pre_trained_weight
save_weight_only=False
save_interval = 1
base_lr = 3e-4
lr_decay = 0.9
nb_epochs = 40
num_thread = 16

# data load
train_data_path = configmanager.train_classifier12_dir
val_data_path = configmanager.val_classifier12_dir
train_batch_size = configmanager.batch_size_classifier12
val_batch_size = 1

if multi_gpus :
    base_lr=base_lr* len(list_gpus)
    train_batch_size = configmanager.batch_size_classifier12 * len(list_gpus)

# train data set
train_datagen = ImageDataGenerator( rescale = 1. / 255)
# width_shift_range=6,
# rotation_range=6)
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    #target_size=(input_size_h, input_size_w),
    target_size=(input_size_w, input_size_h),
    batch_size=train_batch_size,
    color_mode='grayscale',
    shuffle=True,
    # seed=49,
    class_mode='categorical')

# validation data set
val_datagen = ImageDataGenerator(rescale=1. / 255)
# width_shift_range=6,
# rotation_range=6)
val_generator = val_datagen.flow_from_directory(
    val_data_path,
    #target_size=(input_size_h, input_size_w),
    target_size=(input_size_w, input_size_h),
    batch_size=1,
    color_mode='grayscale',
    shuffle=False,
    # seed=49,
    class_mode='categorical')

train_classmap = train_generator.class_indices
val_classmap = val_generator.class_indices

print("train_generator class index: {}".format(train_generator.class_indices))
print("val_generator class index: {}".format(val_generator.class_indices))
nb_classes = len(train_classmap)

class writer:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, text):
        for w in self.writers:
            w.write(text)
    def flush(self):
        pass

def schedule(epoch, decay=0.9):
    lr = base_lr * lr_decay ** (epoch)
    print('##### learning rate of epoch ', int(epoch) + 1, ' : ', lr)
    return lr

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        train_time=time.time() - self.epoch_time_start
        print("Training time:",train_time,"seconds")
        self.times.append(train_time)

inputs = Input(shape=configmanager.input_size_classifier12)

x = InceptionResNetV2(inputs)
x = GlobalMaxPooling2D()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if multi_gpus :
    model_ = Model(inputs=inputs, outputs=predictions)
    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    model = multi_gpu_model(model_, gpus=len(list_gpus))
    model_json = model_.to_json()
    with open(save_dir + '/model_single_gpu.json', "w") as json_file:
        json_file.write(model_json)
else:
    model = Model(inputs=inputs, outputs=predictions)


early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=6,
                               mode='min',
                               verbose=2)

checkpoint = ModelCheckpoint(
        os.path.join(save_dir, 'weights_{epoch:02d}-{val_acc:.4f}.hdf5'),
        monitor='val_acc',
        verbose=2,
        save_best_only=False,
        save_weights_only=save_weight_only,
        mode='max',
        period=save_interval,
        multi_gpus=multi_gpus)

tensorboard = TensorBoard(log_dir=os.path.expanduser(save_dir),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)
time_callback = TimeHistory()
reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
#cbs = [early_stop, checkpoint, reduce_lr, time_callback]
cbs = [checkpoint, reduce_lr, time_callback, tensorboard]
optim = keras.optimizers.Adam(lr=base_lr)

saved = sys.stdout
if pre_trained_weight: #fine-tuning
    print('Fine tuning model from pretrained:',pre_trained_weight)
    model.load_weights(pre_trained_weight)
    base_lr = 1e-4

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
#print(model.summary())
log_file = os.path.join(save_dir, "train.log")
print('Please check output of training process in:', log_file)
f = open(log_file, 'w')
sys.stdout = writer(sys.stdout, f)
# save the classmap in output dir
with open(os.path.join(save_dir, "classmap.json"), 'w') as fd:
    json.dump(train_classmap, fd)
print('train_dir:', train_data_path)
print('val_dir:', val_data_path)

hist = model.fit_generator(
    train_generator,
    verbose=1,
    steps_per_epoch=int(train_generator.samples / train_generator.batch_size),
    epochs=nb_epochs,
    validation_data=val_generator,
    validation_steps=int(val_generator.samples / val_generator.batch_size),
    workers=num_thread,
    callbacks=cbs)

sys.stdout = saved
f.close()
