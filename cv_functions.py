
import cv2
import numpy as np


# Compute runnin average of a sequence of frames
def calculate_background(backgnd, img,weight=0.5):

    if backgnd is None:
        # get image and save as a float, acts as inital value of running average
        backgnd = np.float32(img)  #equivalent = img.copy().astype("float")
        # print("iam here")
        return (backgnd)
        
    # calculate the running average with a given image frame and updat bg variable
    cv2.accumulateWeighted(img, backgnd, weight)
    
    return (backgnd)  
	#weight is Alpha value; determines how fast acumulator modelforgets previous images 
	#[lower aplha value == longer retension, thus sluggish frames. Misses fast movements]
	#lower alpha value effective for backround extraction from a video with fast moving elements



def segment(backgnd,img,thres=25):
    # difference between background and current frame store image in variable d
    d = cv2.absdiff(cv2.convertScaleAbs(backgnd),img)
	# convertScaleAbs changes format/type of accumulated image into usable "uint8"
	# equivalent of foo.astype("uint8")

    thresholdImg = cv2.threshold(d,thres,255,cv2.THRESH_BINARY)[1]
    # _,cnts,_ = cv2.findContours(thresholdImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    return thresholdImg



def findContours(thresholdImg):
    _,cnts,_ = cv2.findContours(thresholdImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts)==0:
        return
    
    else:
        conts = max(cnts,key=cv2.contourArea)
        return conts

