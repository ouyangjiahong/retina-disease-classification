'''
Created by Jiahong Ouyang, 2017/10/31.

Predict images with and without artifacts.
'''

# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
# K.set_image_dim_ordering('tf')

# bs = 4
# test_generator = ImageDataGenerator(rescale = 1./255)
# test_data = test_generator.flow_from_directory(
#         'data/test',
#         target_size = (224, 224),
#         batch_size = bs,
#         class_mode = 'categorical')

# model = load_model('VGG16_trained.h5')
# predict = model.predict_generator(test_data, steps=32)

# print predict

import scipy.misc
import glob
from shutil import copy2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import backend as K
K.set_image_dim_ordering('tf')

num = 909
model = load_model('model_scratch.h5')
dir_path = 'data/'
err_all = 0
acc_all = 0
for i in xrange(4):
	img_paths = dir_path + 'test/r' + str(i) + '/*.jpg'
	# print img_paths
	files = glob.glob(img_paths)
	err = 0
	acc = 0
	for file in files:
		img = scipy.misc.imread(file)
		img = scipy.misc.imresize(img, (224, 224))
		img = img * (1./255)
		img = np.expand_dims(img, axis = 0)
		out = model.predict(img, batch_size = 32, verbose = 0)
		# print out
		if(out[0][i] <= 0.25):
			err = err + 1
			err_all += 1
            # copy2(file, 'data/test/FN')
		else:
			acc = acc + 1
			acc_all += 1
	print "class r" + str(i)
	# print err, acc
	print "true", acc / float(err + acc)
	print "false", err / float(err + acc)
print "overall"
print "accuracy:	" + str(acc_all / float(err_all + acc_all)) 
# img_paths = dir_path + 'test/normal/*.png'
# files = glob.glob(img_paths)
# err = 0
# acc = 0
# for file in files:
# 	img = cv2.imread(file)
# 	img = cv2.resize(img, (224, 224))
# 	img = img * (1./255)
# 	img = np.expand_dims(img, axis = 0)
# 	#print img.shape
# 	out = model.predict(img, batch_size = 32, verbose = 0)
# 	if(out[0][0] > out[0][1] + 0.3):
# 		err = err + 1
# 		# copy2(file, 'data/test/FP')
# 	else:
# 		acc = acc + 1
# print err, acc
# print "false positive", err / float(err + acc)
# print "true negative", acc / float(err + acc)