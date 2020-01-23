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

configfile = 'config/classifier_chinese_config.ini'
configmanager = ConfigManager(configfile)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
# sub_group = configmanager.group_classifier
save_dir = configmanager.save_classifier_dir+ '/train_classifier_chinese_' + training_time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier_train)

input_size_w = configmanager.input_size_classifier[0]
input_size_h = configmanager.input_size_classifier[1]
chanel_size = configmanager.input_size_classifier[2]

save_interval = 1
base_lr = 3e-4
nb_epochs=25
num_thread=10

# data load
train_data_path = configmanager.train_classifier_dir
val_data_path = configmanager.val_classifier_dir

list_gpus = configmanager.gpu_num_classifier_train.split(',')
train_batch_size = configmanager.batch_size_classifier * len(list_gpus)

val_batch_size = configmanager.batch_size_classifier

# train data set
train_datagen = ImageDataGenerator( rescale = 1. / 255)
# width_shift_range=6,
# rotation_range=6)
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(input_size_h, input_size_w),
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
    target_size=(input_size_h, input_size_w),
    batch_size=val_batch_size,
    color_mode='grayscale',
    shuffle=False,
    # seed=49,
    class_mode='categorical')

train_classmap = train_generator.class_indices
val_classmap = val_generator.class_indices

print("train_generator class index: {}".format(train_generator.class_indices))
print("val_generator class index: {}".format(val_generator.class_indices))
nb_classes = len(train_generator.class_indices)

def schedule(epoch, decay=0.9):
    lr = base_lr * 0.9 ** (epoch)
    print('##### learning rate of epoch ', int(epoch) + 1, ' : ', lr)
    return lr


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
f.close()