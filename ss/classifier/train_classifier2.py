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
from config.config_manager import ConfigManager
from model.model_builder import InceptionResNetV2

configfile='config/classifier_config.ini'
configmanager = ConfigManager(configfile)
training_time = datetime.today().strftime('%Y-%m-%d_%H-%M')
#classifier2
sub_group = configmanager.group_classifier2
save_dir = configmanager.save_classifier2_dir+ '/train_classifier2_'+sub_group+'_'+training_time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(configmanager.gpu_num_classifier2_train)

input_size_w = configmanager.input_size_classifier2[0]
input_size_h = configmanager.input_size_classifier2[1]

save_interval = 1
base_lr = 3e-4
lr_decay = 0.9
nb_epochs = 40
num_thread = 16

# data load
train_data_path = configmanager.train_classifier2_dir
val_data_path = configmanager.val_classifier2_dir
train_batch_size = configmanager.batch_size_classifier2
val_batch_size = configmanager.batch_size_classifier2

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

inputs = Input(shape=configmanager.input_size_classifier2)
#inputs = Input(shape=(168,56,1))

x = InceptionResNetV2(inputs)
x = GlobalMaxPooling2D()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

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
        save_weights_only=False,
        mode='max',
        period=save_interval)

reduce_lr = keras.callbacks.LearningRateScheduler(schedule)
#cbs = [early_stop, checkpoint, reduce_lr]
cbs = [checkpoint, reduce_lr]
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

saved = sys.stdout

#print(model.summary())
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
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
