from imutils import paths
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array



def LoadImages(TrainDatatDir,TestDataDir,ImgWidth,ImgHeight):
	"""
	Input Parameter: Given TrainDatatDir and TestDataDir
	Output Parameter: List (X_Train,Y_Train,X_Test，Y_Test) 
	Description:
	The function will take the folder name as the label Name and pre-processing the input image based on the input dimension	
	"""

	TrainImagePaths = sorted(list(paths.list_images(TrainDatatDir)))
	TrainData=[]
	TrainLabel=[]
	for imagePath in TrainImagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (ImgWidth, ImgHeight))
		image = img_to_array(image)
		TrainData.append(image)
		label = imagePath.split(os.path.sep)[-2]
		print("Path is "+imagePath +"label is "+label)
		TrainLabel.append(label)



	TestData=[]
	TestLabel=[]
	TestImagePaths  = sorted(list(paths.list_images(TestDataDir)))
	for imagePath in TestImagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (ImgWidth, ImgHeight))
		image = img_to_array(image)
		TestData.append(image)
		label = imagePath.split(os.path.sep)[-2]
		TrainLabel.append(label)

	return (TrainData,TrainLabel,TestData,TestLabel)

LoadImages("TestImages","TestImages",24,24)










