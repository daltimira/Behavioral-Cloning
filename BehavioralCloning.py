import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from random import shuffle

path_data = './data_sample_save/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2 # correction factor for the steering of the right and left cameras
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # center image
                name = path_data + 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #left image
                name = path_data + 'IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)

                #right image
                name = path_data + 'IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)

                # I will do some image augmentation, will flip the tree images horizontally, and correct the angle

                # center image
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)

                # left image
                images.append(cv2.flip(left_image, 1))
                angles.append(left_angle * -1.0)

                # right image
                images.append(cv2.flip(right_image, 1))
                angles.append(right_angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def main():
    samples = []
    with open(path_data + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = Sequential()
    model.add(Lambda(lambda x:x/255 - 0.5, input_shape=(160,320,3))) # this for normalization, and the '-0.5' is for mean centering the image
    # Add a lambda layer in order to convert to grayscale
    #model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))
    model.add(Cropping2D(cropping=((55,25), (0,0)))) # remove the top 55 pixels and the botton 25 pixels.
    #model.add(Flatten()) #model.add(Flatten(input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='Adam')
    #history_object  = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)

    checkpoint = ModelCheckpoint(filepath="./bestModel/dac_net.h5", monitor='val_loss', save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

    history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 2,  callbacks=[checkpoint, stopper])

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    plt.savefig("meanSquaredErrorLoss.png")

    # Save the model
    model.save("dac_net.h5")

if __name__ == "__main__":
    main()
