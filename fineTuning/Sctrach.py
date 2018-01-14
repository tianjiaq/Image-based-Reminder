import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import LoadImages
def createModel(nClasses):
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128,128,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model


#Create Dataset
(data,labels,CategoryMapping)=LoadImages("/nfs/ug/homes-0/y/yangze3/Image-based-Reminder/images",128,128)
	# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
np.save('Sctrach.npy', CategoryMapping) 
NumberOfClass=len(CategoryMapping)
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

	# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=NumberOfClass)
testY = to_categorical(testY, num_classes=NumberOfClass)




model1 = createModel(NumberOfClass)
batch_size = 256
epochs = 50
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

result = model1.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(testX, testY))
model1.evaluate(testX,testY)
print("[INFO] serializing network...")
model1.save("Sctrach.model")

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), result.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), result.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on InceptionV3")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Sctrach.png")

