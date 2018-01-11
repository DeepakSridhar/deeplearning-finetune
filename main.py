from importlib import reload
import utils; reload(utils)
from utils import *

from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data, save_array, load_array, onehot

from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

path = "C:/Users/deep1/courses/deeplearning1/nbs/data/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

# batch_size=100
batch_size=4

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model

# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)

val_data = get_data(path+'valid')

trn_data = get_data(path+'train')

save_array(model_path+'train_data.bc', trn_data)
save_array(model_path+'valid_data.bc', val_data)

trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)

# trn_features = model.predict(trn_data, batch_size=batch_size)
# val_features = model.predict(val_data, batch_size=batch_size)
#
# save_array(model_path+'train_lastlayer_features.bc', trn_features)
# save_array(model_path+'valid_lastlayer_features.bc', val_features)
#
# trn_features = load_array(model_path+'train_lastlayer_features.bc')
# val_features = load_array(model_path+'valid_lastlayer_features.bc')
#
# # 1000 inputs, since that's the saved features, and 2 outputs, for dog and cat
# lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
# lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
#
# lm.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size,
#        validation_data=(val_features, val_labels))

model.pop()
for layer in model.layers: layer.trainable=False

model.add(Dense(2, activation='softmax'))

gen=image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)

def fit_model(model, batches, val_batches, nb_epoch=1):
    model.fit_generator(batches, samples_per_epoch=batches.n, nb_epoch=nb_epoch,
                        validation_data=val_batches, nb_val_samples=val_batches.n)

opt = Adam(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

fit_model(model, batches, val_batches, nb_epoch=2)

# model.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size,
#        validation_data=(val_features, val_labels))

model.save_weights(model_path+'finetune1.h5')

model.load_weights(model_path+'finetune1.h5')

model.evaluate(val_data, val_labels)

#finetune all dense layers
layers = model.layers
# Get the index of the first dense layer...
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# ...and set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]: layer.trainable=True

K.set_value(opt.lr, 0.01)
fit_model(model, batches, val_batches, 3)

model.save_weights(model_path+'finetune2.h5')

#finetune conv layers
for layer in layers[12:]: layer.trainable=True
K.set_value(opt.lr, 0.001)

fit_model(model, batches, val_batches, 4)

model.save_weights(model_path+'finetune3.h5')

model.load_weights(model_path+'finetune2.h5')
model.evaluate_generator(get_batches(path+'valid', gen, False, batch_size*2), val_batches.n)

