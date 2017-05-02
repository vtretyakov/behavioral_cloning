import csv
import cv2
import numpy as np
import sklearn

#stearing adjustments
correction = 0.2
epochs = 10

samples = []
with open ('training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = 'training/IMG/'+ batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                    if i == 1: #left image
                        angle = float(batch_sample[3]) + correction
                    elif i == 2:#right image
                        angle = float(batch_sample[3]) - correction
                    else:
                        angle = float(batch_sample[3])
                    angles.append(angle)
            
            # trim image to only see section with road
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
        
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#images = []
#measurements = []
#for line in lines:
#    for i in range(3):
#        source_path = line[i]
#        filename = source_path.split('/')[-1]
#        current_path = 'training/IMG/' + filename
#        image = cv2.imread(current_path)
#        images.append(image)
#        if i == 1: #left image
#            measurement = float(line[3]) + correction
#        elif i == 2:#right image
#            measurement = float(line[3]) - correction
#        else:
#            measurement = float(line[3])
#        measurements.append(measurement)

#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs, verbose=1)



history_object = model.fit_generator(train_generator, samples_per_epoch =
                                     len(train_samples), validation_data =
                                     validation_generator,
                                     nb_val_samples = len(validation_samples),
                                     epochs=epochs, verbose=1)

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


