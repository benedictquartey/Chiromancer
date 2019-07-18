import pandas as pd # data analysis toolkit - create, read, update, delete datasets
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np


# load images from disk given filepath
def loadImage(filename):
    return cv2.imread(filename)


def prepareData():
    # read  csv file
    if not os.path.exists("data/IMG/"):
        os.makedirs("data/IMG/")

    data_df = pd.read_csv('data/palm_reader_data.csv',names=['images','labels'])
    
    X,Y=data_df['images'].values,data_df['labels'].values

    images=[]
    
    for i in range(0,len(data_df)):
        X[i]=X[i].strip()
        img_path = X[i]
        img = cv2.imread(img_path)

        images.append(img)
        
    (X_train, X_test, Y_train, Y_test) = train_test_split(images,Y, test_size=0.33, random_state=0)
    return (X_train, X_test, Y_train, Y_test)


# generate multiple training data with varied changes for better learning  (lots of geometric transformations, since images are black and white)
def augment_data(img_filename,label):
    image = loadImage(img_filename) 

    # randomly mirroring image by flipping horizontally
    if np.random.choice(2) == 0:
        image = cv2.flip(image, 1)

# randomly mirroring image by flipping vertically
    if np.random.choice(2) == 0:
        image = cv2.flip(image, 0)

# randomly mirroring image by flipping both vertically and horizontally
    if np.random.choice(2) == 0:
        image = cv2.flip(image, -1)

    # vertical flip
    # translation
    # rotation
    # shear
    # add noise
    

    return image, label



#confirms that all images collected in csv file actually exists in data folder
def data_check():
    if not os.path.exists("data/IMG/"):
        os.makedirs("data/IMG/")
    
    data_df = pd.read_csv('data/palm_reader_data.csv',names=['images', 'label'])
        
    valid_images,invalid_images =0,0
    invalid_filenames = []

    # select only image file names
    img_filenames=data_df['images'].values
    print("\n**************** Beginning Data Verification ****************")
    for filename in img_filenames:
        if isinstance(loadImage(filename), np.ndarray): #check if file loaded from filename is a numpy multidimensional array (images are represented as such)
            valid_images+=1
        else:
            invalid_images+=1
            invalid_filenames.append(filename)
    print("Valid Images: {} || Invalid Images : {}".format(valid_images,invalid_images))
    print("Invalid files are: {}\n".format(invalid_filenames))


# only run if script if invoked directly
if __name__ == "__main__":
    data_check()



