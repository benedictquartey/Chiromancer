import numpy as np           #numpy library for matrix math
import cv2    
import imutils               # basic image processing
import pandas as pd
import os
from datetime import datetime
import time

import cv_functions


data_class=[]
data_images= []
dataSet = {'images':data_images,'label ':data_class}

timeStamp = "0.0.0.0"
backgnd = None
caliberation = False
df=None

def compile_data():
	global df
	df = pd.DataFrame(data=dataSet)
	df =df.drop_duplicates()
	print("Creating data csv file ...")

    # write to csv file without headers and index
	df.to_csv('data/palm_reader_data.csv',index=False,header=False)
	time.sleep(1)
	print("File saved in data/palm_reader_data.csv")


def save_img(frame,data_point_class):
# save training data
	img_data=imutils.resize(frame, width=min(400, frame.shape[1]))
	img_path= "data/IMG/"+datetime.now().strftime('%Y-%m-%d_%H.%M.%S')+".png"
			#save processed image with time stamp

	# create storage folder if it doesnt already exist
	if not os.path.exists("data/IMG/"):
		os.makedirs("data/IMG/")
	
	cv2.imwrite(img_path,img_data)

	data_images.append(img_path)
	data_class.append(data_point_class)

	print("Img Name: {}, Image Class : {}".format(img_path,data_point_class))


def main():
	print("\n**************** Beginning Data Collection ****************\n")
	print(" Collect data: [Option]  C\n View  data:   [Option]  V \n Quit script:  [Option]  Q\n")
	option = input("[Option]: ")

	if (option.lower()=='c'):
		num_data_point = int(input("Number of Datapoints: "))
		data_point_class = input("Class of Datapoints: ")
		collectData(num_data_point,data_point_class)
	
	elif(option.lower()=='v'):
		print("\n**************** View Collected Data  ****************\n")
		viewData()

	elif(option.lower()=='q'):
		print("Quitting ...")
		exit()
		


def collectData(num_data_point,data_point_class):
	global backgnd
	cap = cv2.VideoCapture(0)
	saved_images=0
	frameCount = 0
	
	while(True):
		
		_,frame = cap.read()
		frame = imutils.resize(frame, width=700)
		frame = cv2.flip(frame,1)
		frame_clone = frame.copy()

			# define boundaries of ROI where we look for hand
		roi = frame[10:255,350:590]
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray=cv2.GaussianBlur(gray,(7,7),0)
		# calculate background for the first 30 frames
		if frameCount<30:
			backgnd=cv_functions.calculate_background(backgnd,gray)
			frameCount+=1
			
		# after background calculation, start capture loop until all data is collected
		elif(frameCount==30 and saved_images !=num_data_point):
			threshold_img=cv_functions.segment(backgnd,gray)

			cv2.rectangle(frame_clone,(590,10),(350,255),(0,255,0),2)
			cv2.imshow("Data Collector",frame_clone)
			cv2.imshow("Threshold",threshold_img)
			keypress = cv2.waitKey(100) & 0xFF
			
			if keypress == ord("c"):
				save_img(threshold_img,data_point_class)
				saved_images+=1

			if (saved_images ==num_data_point):
				break

	cap.release()
	cv2.destroyAllWindows()
	cv2.waitKey(1) #extra waitkey to ensure visualization window closes

	
	compile_data()
	main()

def viewData():
	print(df)
	main()

			

	


# only run if script if invoked directly
if __name__ == "__main__": 
    main()
