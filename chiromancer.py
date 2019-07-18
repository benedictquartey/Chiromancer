
# @Author Benedict Quartey


#import dependencies
import numpy as np           #numpy library for matrix math
import cv2                      #opencv library
import imutils               # basic image processing
import socket
import keras.models
from keras.models import model_from_json
import tensorflow as tf
import cv_functions
import sys
from keras.models import load_model


backgnd = None
UDP_IP = "127.0.0.1"
UDP_PORT = 5065


# global model, graph

print ("UDP target IP:", UDP_IP)
print ("UDP target port:", UDP_PORT)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP


def sendToSim(data):
	my_str_as_bytes = str.encode(data)
	sock.sendto((my_str_as_bytes ) , (UDP_IP, UDP_PORT))
	print("Sent ",data, "to Unity Simulation")


def loadModel():
	loaded_model = load_model(sys.argv[1])


	return loaded_model



def predict(img):
	cv2.imwrite("temp.png",img) #temporary solution to gain additional rgb channel dimemsion that is striped when image becomes thresholded
	img=cv2.imread("temp.png")
	
	x = []
	x.append(img)
	x=np.array(x)
	x= x.astype('float32')
	print(x.shape)

	out = model.predict(x, batch_size=1)
	print("Class Probabilities: {}".format(out))

	print("Prediction: {}".format(np.argmax(out)))
	
	#convert the response to a string
	response = np.array_str(np.argmax(out,axis=1))
	print(response)
	sendToSim(response[1])
		


def main():
	global backgnd
	cap=cv2.VideoCapture(0)
	frameCount = 0

	while(True):
		_,frame = cap.read()

		frame = imutils.resize(frame, width=700)
		frame = cv2.flip(frame,1)
		frameCl = frame.copy()

		# define boundaries of ROI where we look for hand
		roi = frame[10:255,350:590]

		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray=cv2.GaussianBlur(gray,(7,7),0)

		if frameCount<30:
			backgnd=cv_functions.calculate_background(backgnd,gray)
			
		else:
			thresholdImg = cv_functions.segment(backgnd,gray)
			hand_contour = cv_functions.findContours(thresholdImg)
			if hand_contour is not None:

				cv2.drawContours(frameCl,[hand_contour + (350,10)],-1,(0,0,255))
				cv2.imshow("Threshold", thresholdImg)

				keypress = cv2.waitKey(100) & 0xFF
				if keypress == ord("p"):
					predict(thresholdImg)

		cv2.rectangle(frameCl,(590,10),(350,255),(0,255,0),2)
		frameCount+=1

		cv2.imshow("Feed",frameCl)
		

	cap.release()
	cv2.destroyAllWindows() 


# only run if script if invoked directly
if __name__ == "__main__":
	model = loadModel()
	# predict("test")
	main()



