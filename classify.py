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
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')

# build the model
# use VGG16 as the base model and add 3 fully-connected layer on the top
# keep the first 15 layers and train the last conv block of base model and the top 3 fully-connected layer
# use gradient descent with momentum as optimizer, the cross entropy as loss

# base_model
base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
# base_model = base_model[:11]

while len(base_model.layers) > 7:
    base_model.layers.pop()
    
base_model.layers[-1].outbound_nodes = []

#add higher layers
base_model.outputs = [base_model.layers[-1].output]
x = base_model.get_layer('block2_pool').output

#block 3
x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

#block 4
x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

#block 5
x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, (3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2))(x)

x = Flatten(input_shape = base_model.output_shape[1:])(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(4, activation = 'softmax')(x)

model = Model(inputs = base_model.input, output = prediction)

plot_model(model, to_file='model.png')

for layer in model.layers[:7]: #only train last two block of conv & dense
  layer.trainable = False
for layer in model.layers[7:]:
  layer.trainable = True

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

model_name = 'VGG16_trained.h5'
model.save(model_name)
print('Saved trained model')
print('training time:'+ str(end - start))


evaluation = model.evaluate_generator(test_data, steps=32) #32
print('Model Accuracy = %.2f' % (evaluation[1]))

#predict = model.predict_generator(test_data, steps=32)



