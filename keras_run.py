import csv
import numpy as np
import pandas as pd
import cv2
import pdb
import os
import csv
import glob
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from PIL import Image, ImageOps
from scipy import ndimage

from keras_model import get_nvidia_model

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--l2reg', default=0.0, type=float)
parser.add_argument('--loss', default='l2')
parser.add_argument('--direction', default='center')
parser.add_argument('--train_dir', default='Ch2/')
parser.add_argument('--val_random', action='store_true')
parser.add_argument('--drop_low_angles', action='store_true')
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--steer_threshold', default=0.03, type=float)
parser.add_argument('--steer_correction', default=0.1, type=float)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_samples', default=-1, type=int)
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--root_dir', default='/Users/buku/work/cs231n/project')
parser.add_argument('--pretrained', default='None')
parser.add_argument('--save_model', default='None')
parser.add_argument('--output', default='angle')

img_shape = (640, 480)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# Save histogram of steering angles
def save_hist(data, name):
    plt.figure()
    plt.hist(data, bins=20, color='green')
    plt.xlabel('Steering angles')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of '+name.replace("_"," "))
    plt.savefig(name+".png")

def preprocess(samples, steer_threshold, drop_low_angles):
    samples = np.array(samples, dtype = object)
    print("Number of samples before dropping low steering angles: {}".format(samples.shape[0]))
    index = np.where( (np.abs(samples[:,1]) < steer_threshold) == True)[0]
    if drop_low_angles == False:
        rows = [i for i in index if np.random.randint(10) < 9]
    else:
        rows = index
    samples = np.delete(samples, rows, 0)
    print("Removed %s rows with low steering"%(len(rows)))
    print("Number of samples after dropping low steering angles: {}".format(samples.shape[0]))

    return samples

def correct_steering(angle, ix, steer_correction):
    if ix == 1: #left
        return angle + steer_correction
    elif ix == 2: #right
        return angle - steer_correction
    else:
        return angle

def make_dataset(dir, direction, train, steer_threshold, drop_low_angles, steer_correction):
    images = []
    root = os.path.join(dir, 'data/')

    pf = 'train' if train == True else 'val'
    cf = "data/{}_center*.csv".format(pf)
    lf = "data/{}_left*.csv".format(pf)
    rf = "data/{}_right*.csv".format(pf)

    c_files = []
    l_files = []
    r_files = []
    if direction == 'center':
        c_files = [f for f in glob.glob(cf)]
    elif direction == 'left':
        l_files = [f for f in glob.glob(lf)]
    elif direction == 'right':
        r_files = [f for f in glob.glob(rf)]
    else:
        c_files = [f for f in glob.glob(cf)]
        l_files = [f for f in glob.glob(lf)]
        r_files = [f for f in glob.glob(rf)]
        print("Gathering all files")

    for ix, files in enumerate([c_files, l_files, r_files]):
        for f in files:
            df = pd.read_csv(f, dtype={'angle': np.float,'torque': np.float, 'speed': np.float})
            for row in df.iterrows():
                angle = row[1]['angle']
                speed = row[1]['speed']
                torque = row[1]['torque']
                angle = correct_steering(angle, ix, steer_correction)
                fname = "Ch2/{}".format(row[1]['filename'])

                if is_image_file(fname):
                    path = os.path.join(dir, fname)
                    item = (path, angle, speed, torque, ix, 0)
                    images.append(item)
                    item = (path, -1*angle, speed, torque, ix, 1)
                    images.append(item)
    images = preprocess(images, steer_threshold, drop_low_angles)
    return images

def make_harddataset(dir, direction, train, steer_threshold, drop_low_angles, steer_correction):
    images = []

    df = pd.read_csv("hardexamples.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = row[1]['fnames']
        if is_image_file(fname):
            item = (fname, angle, 0, 0, 0, 0)
            images.append(item)
    images = np.array(images)
    print("Number of hard data examples", images.shape)
    return images

def split_dataset(dir, direction, steer_threshold, drop_low_angles, steer_correction):
    images = []
    with open('Ch2/interpolated.csv') as csvfile:
        if direction == 'center':
            _directions = ['center']
        elif direction == 'left':
            _directions = ['left']
        elif direction == 'left':
            _directions = ['right']
        else:
            _directions = ['center', 'left', 'right']

        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            d = row[5].split('/')[0]
            path = 'Ch2/' + d + '/' + row[5].split('/')[1]
            angle = float(row[6])
            torque =  float(row[7])
            speed = float(row[8])

            if d in _directions:
                if d == 'center':
                    ix = 0
                elif d == 'left':
                    ix = 1
                else:
                    ix = 2
                angle = correct_steering(angle, ix, steer_correction)
                item = (path, angle, speed, torque, ix, 0)
                images.append(item)
                item = (path, -1*angle, speed, torque, ix, 1)
                images.append(item)

    images = preprocess(images, steer_threshold, drop_low_angles)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2)

    return train_imgs, val_imgs

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    cols, rows = img_shape
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr, steer_ang

def augment_image(im, angle):
    im = augment_brightness(im)
    im, angle = trans_image(im, angle, 50)
    return im, angle

def target_normalization(imgs):

    print("Angles Before Normalization: ", np.mean(imgs[:,1]),  np.std(imgs[:,1]))
    print("Speed Before Normalization: ", np.mean(imgs[:,2]),  np.std(imgs[:,2]))
    print("Torque Before Normalization: ", np.mean(imgs[:,3]),  np.std(imgs[:,3]))

    means = [-1.5029124843010954e-05, 15.280215684662938, -0.09274196373527277]
    stds = [0.338248459692, 5.5285360815, 0.905531102853]
    imgs[:,1] = (imgs[:,1]-means[0])/stds[0]
    imgs[:,2] = (imgs[:,2]-means[1])/stds[1]
    imgs[:,3] = (imgs[:,3]-means[2])/stds[2]

    return imgs

def generator(samples, batch_size, augmentation=False, output='angle'):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+batch_size]

            images = []
            y = []
            for bs in batch_samples:
                img_name  = bs[0]
                steering = bs[1]
                speed = bs[2]
                torque = bs[3]
                flip = bs[5]
                img = pil_loader(img_name)
                img = img if flip==0 else img.transpose(Image.FLIP_LEFT_RIGHT)  #cv2.flip(img,1)# Flip image if flip was 1
                img = np.asarray(img)

                if augmentation:
                    img, steering = augment_image(img, steering)

                if output != 'angle':
                    y.append([steering, speed, torque])
                else:
                    y.append(steering)

                images.append(img)

            X_train = np.array(images)
            y_train = np.array(y)
            yield shuffle(X_train, y_train)

def val_output(s):
    img_name  = s[0]
    steering = s[1]
    img = pil_loader(img_name)
    img = np.asarray(img)
    return img, steering

def angle_loss(y_true, y_pred):
    return K.mean((y_pred[:,0]-y_true[:,0])**2)

def main(args):

    if args.val_random == False:
        train_imgs = make_harddataset(args.root_dir, args.direction,
                                        True, args.steer_threshold, args.drop_low_angles, args.steer_correction)
        val_imgs = make_dataset(args.root_dir, args.direction, False, 0, False, args.steer_correction)
    else:
        train_imgs, val_imgs = split_dataset(args.root_dir, args.direction,
                                        args.steer_threshold, args.drop_low_angles, args.steer_correction)

    if args.num_samples > 0:
        train_imgs = train_imgs[:args.num_samples]
        val_imgs = val_imgs[:1000]

    if args.output != 'angle':
        train_imgs = target_normalization(train_imgs)
        val_imgs = target_normalization(val_imgs)

    train_gen = generator(train_imgs, args.batch_size, args.augmentation, args.output)
    val_gen = generator(val_imgs, args.batch_size, False, args.output)

    num_outputs = 1 if args.output == 'angle' else 3
    model = get_nvidia_model(num_outputs, args.l2reg)

    if args.pretrained != 'None':
        model.load_weights("{}.h5".format(args.pretrained))
        #print(model.get_weights())

    pred = []
    real = []
    df = pd.read_csv("Ch2/interpolated.csv", dtype={'angle': np.float,'torque': np.float, 'speed': np.float})
    df = df[df['frame_id']=='center_camera'].reset_index(drop=True)

    fnames = np.array('Ch2/'+ df['filename'])
    angles = np.array(df['angle'])
    for ix,tr in enumerate(zip(fnames, angles)):
        #if ix >= 100:
        #    break
        img, angle = val_output(tr)
        real.append(angle)
        pred.append(model.predict(img.reshape(-1, 480, 640, 3))[0][0])


    plt.figure(figsize=(16,9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig('pred_lookahead.png')
    #plt.show()

    df = pd.DataFrame()
    df['fnames'] = fnames
    df['angle'] =  real
    df['pred'] = pred
    df.to_csv('results_lookahead.csv')

    return



    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer=adam, metrics=[angle_loss])

    #callbacks = [EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
    if args.save_model != 'None':
        filepath=args.save_model + "-{val_angle_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='angle_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
    else:
        callbacks_list = []
    history_object = model.fit_generator(train_gen,
                    steps_per_epoch = (len(train_imgs)/args.batch_size),
                    validation_data = val_gen,
                    validation_steps = (len(val_imgs)/args.batch_size),
                    epochs = args.num_epochs,
                    verbose=1,
    				callbacks=callbacks_list)

    if args.save_model != 'None':
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("{}.h5".format(args.save_model))

    ### plot the training and validation loss for each epoch
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

if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
  with K.get_session():
      main(args)
