#adapted from PyimageSearch for testing accuracy between my standard NN_model architecture and a popular model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
import data_processing

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import keras
from keras.callbacks import ModelCheckpoint


num_classes = 6

# seeding to enable exact reproduction of learning results
np.random.seed(0)

def preprocess():
	(X_train, X_test, Y_train, Y_test)=data_processing.prepareData()
	x_train = np.array(X_train)
	y_train = np.array(Y_train)
	x_test = np.array(X_test)
	y_test = np.array(Y_test)

#more reshaping
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return(x_train,x_test,y_train,y_test)

def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)



# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object
valAug = ImageDataGenerator()




mean = np.array([123.68, 116.779, 103.939], dtype="float32")

trainAug.mean = mean
valAug.mean = mean

(x_train, x_test, y_train, y_test)=preprocess()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# initialize the training generator
trainGen = trainAug.flow(x_train,y_train,batch_size=32)

 
# initialize the validation generator
valGen = valAug.flow(x_test,y_test,batch_size=32)


# CNN surgury
# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(245, 240, 3)))
 
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation="softmax")(headModel)
 
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)



# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False



# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=10 ,
	validation_data=valGen,
	validation_steps=30,
	epochs=20,
	verbose=1)
 
# reset the testing generator and evaluate the network after
# fine-tuning just the network head
print("[INFO] evaluating after fine-tuning network head...")
valGen.reset()


plot_training(H, 20, 'head_only_plot' )



#now to unfreeze some of the latter conv layers
# reset our data generators
trainGen.reset()
valGen.reset()

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
	layer.trainable = True

# loop over the layers in the model and show which ones are trainable
# or not
for layer in baseModel.layers:
	print("{}: {}".format(layer, layer.trainable))



# callback function to be executed after every training epoch, only saves the trsined model
# if its validation mean_squared_error is less than the model from the previoud epoch
interimModelPoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only = 'true',
                                    mode = 'auto')

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers

print("[INFO] training head and latter conv...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=10 ,
	validation_data=valGen,
	validation_steps=30,
	epochs=10,
	callbacks = [interimModelPoint],
	verbose=1)


plot_training(H, 10, 'full_training_plot' )



# # serialize the model to disk
# print("[INFO] serializing network...")
# model.save('models/model.h5')
