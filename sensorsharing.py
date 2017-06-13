import csv
import cv2
import numpy as np
import pdb
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.layers.pooling import MaxPooling2D

import pandas as pd


# Data Path relative to your machine
DATA_PATH = "Ch2/"

# Hyperparameters
STEER_CORRECTION = 0.1  # Using a left and right image steer correction
BATCH_SIZE       = 64
STEER_THRESHOLD  = 0.03 # 80% images with steering angle less than threshold
EPOCHS           = 25   # Maximum number of EPOCHS. (Early stopping)

LOOKAHEAD = 20

val_frac = 0.2

# Data Preprocessing and Augmentation
samples =[]
str_angle_orig = []
str_angle_aug = []
str_angle_drp = []

with open(DATA_PATH+'interpolated.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[5].split('/')[0] == 'center':
            centername = DATA_PATH+row[5]
            steering_center = float(row[6])
        #if row[5].split('/')[0] == 'left':
        #    steering_center += STEER_CORRECTION
        #if row[5].split('/')[0] == 'right':
        #    steering_center -= STEER_CORRECTION
            str_angle_orig.append(steering_center)
            def append_samples(name, angle, flip):
                samples.append([name, angle, flip])
                str_angle_aug.append(angle)

            append_samples(centername,steering_center, 0)
            #append_samples(centername,-steering_center,1)

samples = np.array(samples, dtype = object)

# Reduce data with low steering angles: Drop rows with probability 0.8
print("Number of samples before dropping low steering angles: {}".format(samples.shape[0]))
#index = np.where((np.abs(samples[:,1])<STEER_THRESHOLD)==True)[0]
#rows = [i for i in index if np.random.randint(10) < 8]
#samples = np.delete(samples, rows, 0)
#print("Removed %s rows with low steering"%(len(rows)))
print("Number of samples after dropping low steering angles: {}".format(samples.shape[0]))

#for row in samples:
#    str_angle_drp.append(row[1])

# Save histogram of steering angles
def save_hist(data, name):
    plt.figure()
    plt.hist(data, bins=20, color='green')
    plt.xlabel('Steering angles')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of '+name.replace("_"," "))
    plt.savefig(name+".png")

#save_hist(str_angle_orig, "steering_angles_original")
#save_hist(str_angle_aug,  "steering_angles_after_augmentation")
#save_hist(str_angle_drp,  "steering_after_dropping_low_angles")

#train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Generator for Keras model.fit
def generator(samples, batch_size, t):
    num_samples = len(samples)
    if t:
        a = 0#int(val_frac*num_samples)
        b = int((1-val_frac)*num_samples)#num_samples
    else:
        a = int((1-val_frac)*num_samples)
        b = num_samples
    while True: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(a, b, batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            i = 0
            for batch_sample in batch_samples:
                img_name  = batch_sample[0]
                steering = batch_sample[1]
                flip     = batch_sample[2]
                if offset+i+LOOKAHEAD<b:
                    x = LOOKAHEAD
                else:
                    x = 0
                img = cv2.imread(img_name) - cv2.imread(samples[offset+i+x][0])
                #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
                #h_channel = hls[:,:,0]
                #a, b = h_channel.shape
                #hls1 = np.reshape(h_channel, (a,b,1))
                images.append(img)
                angles.append(steering)
                i = i+1

            X_train = np.array(images)
            y_train = np.array(angles)
            # print("X_train.shape {}".format(X_train.shape))
            yield X_train, y_train

# compile and train the model using the generator function
train_generator      = generator(samples, BATCH_SIZE, True)
validation_generator = generator(samples, BATCH_SIZE, False)

# Keras model
model =  Sequential()
model.add(Lambda(lambda x : x/255, input_shape=(480,640,3)))
model.add(Cropping2D(cropping=((240,0),(0,0)))) # trim image to only see section with road

model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dense(291))
#model.add(Dropout(0.2))
#model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))


model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
#model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
#model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
#model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss="mse", optimizer=adam)
#model.compile(loss="mse", optimizer="adam")
#callbacks = [EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
#history_object = model.fit_generator(train_generator, steps_per_epoch = (int(len(samples)*(1-val_frac))/BATCH_SIZE), validation_data=validation_generator, \
#                validation_steps = (int(val_frac*len(samples))/BATCH_SIZE), epochs=EPOCHS, verbose = 1)
				#callbacks=callbacks)

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model into h5 file")

#model.save("model.h5")

### plot the training and validation loss for each epoch
"""
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
print(history_object.history['val_loss'])
print(history_object.history['loss'])
"""

model.load_weights("gaurav.h5")

def full_generator(samples, batch_size, t):
    num_samples = len(samples)
    a = 0
    b = int(num_samples)
    while True: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(a, b, batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            i = 0
            for batch_sample in batch_samples:
                img_name  = batch_sample[0]
                steering = batch_sample[1]
                flip     = batch_sample[2]
                if offset+i+LOOKAHEAD<b:
                    x = LOOKAHEAD
                else:
                    x = 0
                img = cv2.imread(img_name) - cv2.imread(samples[offset+i+x][0])
                #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
                #h_channel = hls[:,:,0]
                #a, b = h_channel.shape
                #hls1 = np.reshape(h_channel, (a,b,1))
                images.append(img)
                angles.append(steering)
                i = i+1

            X_train = np.array(images)
            y_train = np.array(angles)
            # print("X_train.shape {}".format(X_train.shape))
            yield X_train, y_train

full_gen = full_generator(samples, BATCH_SIZE, True)

print("Full imgs shape: ", samples.shape)
pred = []
real = []
for ix in range(int(len(samples)/BATCH_SIZE)):
    inp, _real = next(full_gen)
    _pred = model.predict(inp)
    #print(_pred[:,0].shape, _real.shape)
    real.extend(_real)
    pred.extend(_pred[:,0])

pred = np.array(pred)
real = np.array(real)
#print(pred, real)
print("Mean Error: ", np.sqrt(np.mean((pred-real)**2)) )

plt.figure(figsize=(16,9))
plt.plot(pred, label='Predicted')
plt.plot(real, label='Actual')
plt.legend()
plt.savefig('pred_baseline.png')

df = pd.DataFrame()
df['angle'] =  real
df['pred'] = pred
df.to_csv('results_baseline.csv')
