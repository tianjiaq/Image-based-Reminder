
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import argparse
import imutils
import cv2
from utils import GetLabelAndCategory
from imutils import paths 
print("[INFO] loading network...")
model = load_model("Sctrach.model")
TrainDatatDir ="TestImages"
TrainImagePaths = sorted(list(paths.list_images(TrainDatatDir)))
for img_path in TrainImagePaths:
	#img_path = 'TestImages/tomato2.jpeg'
	img = image.load_img(img_path, target_size=(128, 128))
	x = image.img_to_array(img)
	x = np.array(x, dtype="float") / 255.0
	x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)

	read_dictionary = np.load('Sctrach.npy').item()
	Mapping= GetLabelAndCategory(read_dictionary)

	preds = model.predict(x)
	print (preds)
	it = np.nditer(preds, flags=['f_index'])
	result=[]
	while not it.finished:
		result.append(it[0])
		it.iternext()
	max_probality_Catgory=max(result)
	index=result.index(max_probality_Catgory)
	Predicted_Category= Mapping[index]
	orig = cv2.imread(img_path)
	orig = cv2.resize(orig, (400, 400))

	label = "{}: {:.2f}%".format(Predicted_Category, max_probality_Catgory * 100)
	 
	# draw the label on the image
	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	#path ="%s.png" %(img_path)
	#print(path)
	#cv2.imwrite("%s.png" %(img_path), output)
	 
	# show the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)





