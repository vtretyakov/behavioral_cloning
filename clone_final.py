import csv
import cv2
import numpy as np

### stearing adjustments
correction = 0.2

### number of training epochs
epochs = 10

### open the CSV file containing pathes to training set images
lines = []
with open ('training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

### use images from the left and right cameras and add corrections to the steering angle
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'training/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        if i == 1: #left image
            measurement = float(line[3]) + correction
        elif i == 2:#right image
            measurement = float(line[3]) - correction
        else:
            measurement = float(line[3])
        measurements.append(measurement)

### augment images by horizontal flipping
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

### convert images to numpy array for used with Keras
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

### create a model based on paper from Nvidia (M.Bojarski et al., "End to End Learning for Self-Driving Cars")
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

### use Mean Squared Error and Adam optimizer for training
model.compile(loss='mse', optimizer='adam')

### split the training set to have 20% of validation data, enable data shuffling, train and store training history
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, verbose=1)

### save the trained model into file
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


