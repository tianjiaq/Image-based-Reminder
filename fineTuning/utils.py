from imutils import paths
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
import random

def Category_To_Int(CategoryAndLabel,label):
	return CategoryAndLabel[label]

def Listdir_not_Hidden(path):
	Category=[]
	for f in os.listdir(path):
		if not f.startswith('.'):
			Category.append(f)
	return Category 

def GetCategoryAndLabel(base_path): 
	CategoryAndLabel={}
	for index,item in enumerate(Listdir_not_Hidden(base_path)):
		CategoryAndLabel[item]=index
	return CategoryAndLabel

def GetLabelAndCategory(Category):
	LabelAndCategoryMapping={}
	for key, value in Category.items():
	 	LabelAndCategoryMapping[value]=key
	return LabelAndCategoryMapping
	 	



def LoadImages(TrainDatatDir,ImgWidth,ImgHeight,TestDataDir=None):
	"""
	Input Parameter: Given TrainDatatDir and TestDataDir
	Output Parameter: List (X_Train,Y_Train,Category) 
	Description:
	The function will take the folder name as the label Name and pre-processing the input image based on the input dimension	
	"""
	Category= GetCategoryAndLabel(TrainDatatDir)
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
		Label_index=Category_To_Int(Category,label)
		TrainLabel.append(Label_index)


	if TestDataDir==None:
		print ("No TrainData")
		return (TrainData,TrainLabel,Category)
	
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

#LoadImages("/nfs/ug/homes-0/y/yangze3/Image-based-Reminder/images",1,1)









