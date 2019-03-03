import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Dropout
from random import shuffle

#path_data = './data/data/'
#path_data = './data_t1_moveCenterOnCurve/'
#path_data = './data_t1_2/'
path_data = './data/'
old_model = "./dac_net_v8_regularization07_epoch5_data_gray.h5"
new_model = "./dac_net_v20_regularization05_once_epoch5_data.h5"

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.1 # correction factor for the steering of the right and left cameras
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
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # left image
                name = path_data + 'IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)

                #right image
                name = path_data + 'IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)

                # I will do some image augmentation, will flip the tree images horizontally, and correct the angle
                # we only flip images if we are in a curve (center angles is different zero) in order to help data balance.
                # Flipping image when there is not a curve, might inbalance more the data, and the model is not helping learning something new.
                if (center_angle != 0):
                    # center image
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)

                    # left image
                    images.append(cv2.flip(left_image, 1))
                    angles.append(left_angle * -1.0)

                    # right image
                    images.append(cv2.flip(right_image, 1))
                    angles.append(right_angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def main():
    samples = []
    with open(path_data + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    print("num samples data all!!!!: ")
    print(len(samples))

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=16)
    validation_generator = generator(validation_samples, batch_size=16)

    if not os.path.isfile(old_model):
        print("Creating a new model")

        model = Sequential()
        # Add a lambda layer in order to convert to grayscale
        # model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(160,320,3)))

        #model.add(Lambda(lambda x:x/255 - 0.5, input_shape=(160,320,3))) # this for normalization, and the '-0.5' is for mean centering the image
        model.add(Lambda(lambda x:x/127.5 - 1.0, input_shape=(160,320,3)))
        #model.add(Lambda(lambda x:x/255 - 0.5))
        model.add(Lambda(lambda x: tf.image.resize_images(x, size=[80,160])))
        #model.add(Cropping2D(cropping=((55,25), (0,0)))) # remove the top 55 pixels and the botton 25 pixels.
        model.add(Cropping2D(cropping=((25,12), (0,0))))
        #model.add(Flatten()) #model.add(Flatten(input_shape=(160,320,3)))

        # changing stride to (1,1) as we have reduces the input image by 2
        model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding = 'valid', activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2), padding='valid', activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Conv2D(48,kernel_size=(5,5),strides=(1,1), padding='valid', activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Conv2D(64, kernel_size=(3,3),strides=(1,1), padding='valid', activation="relu"))
        #model.add(Dropout(0.2))
        #model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='valid', activation="relu"))

        model.add(Flatten())

        model.add(Dropout(0.5))

        model.add(Dense(100))

        model.add(Dropout(0.5))

        model.add(Dense(50))
        #model.add(Dropout(0.2))
        model.add(Dense(10))
        #model.add(Dropout(0.7))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='Adam')
    else:
        print("Loading model")
        model = load_model(old_model,  custom_objects={"tf": tf})

    #history_object  = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)

    checkpoint = ModelCheckpoint(filepath=new_model, monitor='val_loss', save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

    history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 2, callbacks=[checkpoint, stopper])

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    plt.savefig("meanSquaredErrorLoss_v20_regularization05_epoch5_data.png")

    # Save the model
    #model.save("dac_net_v1_epoch1_sim.h5")

if __name__ == "__main__":
    main()
