from imutils import paths
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
import random

def Category_To_Int(label):
	if label =="yellow_banana":
		result=0
	elif label=="tomato":
		result=1
	elif label=="pumpkin":
		result=2
	return result



def LoadImages(TrainDatatDir,ImgWidth,ImgHeight,TestDataDir=None):
	"""
	Input Parameter: Given TrainDatatDir and TestDataDir
	Output Parameter: List (X_Train,Y_Train,X_Test，Y_Test) 
	Description:
	The function will take the folder name as the label Name and pre-processing the input image based on the input dimension	
	"""

	TrainImagePaths = sorted(list(paths.list_images(TrainDatatDir)))
	random.seed(42)
	random.shuffle(TrainImagePaths)
	TrainData=[]
	TrainLabel=[]
	for imagePath in TrainImagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (ImgWidth, ImgHeight))
		image = img_to_array(image)
		TrainData.append(image)
		label = imagePath.split(os.path.sep)[-2]
		#print("Path is "+imagePath +"label is "+label)

		TrainLabel.append(Category_To_Int(label))


	if TestDataDir==None:
		print ("No TrainData")
		return (TrainData,TrainLabel)
	
	TestData=[]
	TestLabel=[]
	TestImagePaths  = sorted(list(paths.list_images(TestDataDir)))
	random.seed(42)
	random.shuffle(TestImagePaths)
	for imagePath in TestImagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (ImgWidth, ImgHeight))
		image = img_to_array(image)
		TestData.append(image)
		label = imagePath.split(os.path.sep)[-2]
		TrainLabel.append(Category_To_Int(label))

	return (TrainData,TrainLabel,TestData,TestLabel)

LoadImages("/Users/zeyang/Documents/GitHub/Image-based-Reminder-group/images",ImgWidth=24,ImgHeight=24)









