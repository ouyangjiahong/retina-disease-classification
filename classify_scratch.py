'''
Created by Jiahong Ouyang, 2018/02/13.

Classification of retina images of four different severity level.
Fine-tune on VGG16
'''

import time
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.engine.topology import Input
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')

# build the model from the scratch
input_img = Input((224, 224, 3))

#block 1
x = ZeroPadding2D((1,1))(input_img)
x = Conv2D(16, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(16, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

#block 2
x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

#block 3
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

# fully-connected layers
x = Flatten()(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(4, activation = 'softmax')(x)

model = Model(inputs = input_img, output = prediction)

plot_model(model, to_file='model_scratch.png')

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.Adam(lr = 0.0001),
              metrics = ['accuracy'])

# data augmentation
# random rotate in -180~180(because of the symmetry), little whitening, rescale to [0,1]
bs = 64
train_generator = ImageDataGenerator(
        rotation_range = 180,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'constant',
        cval = 0,
        rescale = 1./255)

val_generator = ImageDataGenerator(rescale = 1./255)

# no augmentation for test
test_generator = ImageDataGenerator(rescale = 1./255)

train_data = train_generator.flow_from_directory(
        'data/train',
        target_size = (224, 224),
        classes = ['r0', 'r1', 'r2', 'r3'],
        batch_size = bs,
        shuffle = True,
        class_mode = 'categorical')

val_data = val_generator.flow_from_directory(
        'data/test',
        target_size = (224, 224),
        classes = ['r0', 'r1', 'r2', 'r3'],
        batch_size = bs,
        class_mode = 'categorical')

test_data = test_generator.flow_from_directory(
        'data/test',
        target_size = (224, 224),
        classes = ['r0', 'r1', 'r2', 'r3'],
        batch_size = bs,
        shuffle = False,
        class_mode = 'categorical')

#train
checkpoint = ModelCheckpoint(
		'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
		monitor = 'val_acc',
		save_best_only = True,
		mode = 'max',
		period = 5)

start = time.clock()
model.fit_generator(
        train_data,
        steps_per_epoch = 400,  
        epochs = 100,
        callbacks = [checkpoint], 
        validation_data = val_data,
        validation_steps = 20,
        shuffle = True) 
 
end = time.clock()

model_name = 'model_scratch.h5'
model.save(model_name)
print('Saved trained model')
print('training time:'+ str(end - start))


evaluation = model.evaluate_generator(test_data, steps=32) #32
print('Model Accuracy = %.2f' % (evaluation[1]))

#predict = model.predict_generator(test_data, steps=32)



