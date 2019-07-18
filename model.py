


from keras.models import Sequential
#dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
#into respective layers
from keras.layers import Dense, Dropout, Flatten
#for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.vgg16 import VGG16

from keras.models import load_model,model_from_json
import sys

num_classes = 6
imageHeight, imageWidth, imageChannels = 245,240,3

# for some reason outperforms finetuned VGG16 model in practice
def nn_model():  
    #build  model
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=(imageHeight, imageWidth, imageChannels)))
	#again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #choose the best unique features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
	#randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
	#flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
	#fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
	#one more dropout for convergence' sake :)  
    model.add(Dropout(0.5))
	#output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_classes, activation='softmax'))

    return model

def eval_model(model,x_test,y_test):
    loss,accuracy = model.evaluate(x_test,y_test)
    print('loss:', loss)
    print('accuracy:', accuracy)


def build_model_from_json():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
    loaded_model.load_weights("model.h5")  


# model = VGG16(weights='imagenet', include_top=False)
# model = load_model(sys.argv[1])


# model.summary()
