
# @Author Benedict Quartey

import matplotlib.pyplot as plt
import numpy as np #matrix math
#simplified interface for building models 
import keras
from keras.callbacks import ModelCheckpoint
import model as NN_model
import data_processing

#for reading files
import os



batch_size = 128

num_classes = 6
epochs = 10

# inputshape for images
imageHeight, imageWidth, imageChannels = 245,240,3


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

def plot_training(model_history, Num_epochs, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, Num_epochs), model_history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, Num_epochs), model_history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, Num_epochs), model_history.history["acc"], label="train_acc")
	plt.plot(np.arange(0, Num_epochs), model_history.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def train():
	(x_train,x_test,y_train,y_test) = preprocess()

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	#get  model
	model = NN_model.nn_model()
	model.summary()
	# callback function to be executed after every training epoch, only saves the trsined model
	# # if its validation mean_squared_error is less than the model from the previoud epoch
	interimModelPoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only = 'true',
                                    mode = 'auto')



	#Adaptive learning rate (adaDelta) is a popular form of gradient descent 
	#categorical cross entropy since we have multiple classes (10) 
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])

	#train!
	history=model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test),
			  callbacks=interimModelPoint)
	 #performance evaluation 
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	plot_training(history,epochs,"model_performance")


		
# only run if script if invoked directly
if __name__ == "__main__": 
    train()

