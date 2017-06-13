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
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from PIL import Image, ImageOps
from scipy import ndimage

from keras_model import get_nvidia_model, get_lstm_model

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
parser.add_argument('--window_len', default=4, type=int)
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--root_dir', default='/Users/buku/work/cs231n/project')
parser.add_argument('--pretrained', default='None')
parser.add_argument('--save_model', default='None')
parser.add_argument('--output', default='angle')
parser.add_argument('--mode', default='concat')
parser.add_argument('--lookahead', action='store_true')
parser.add_argument('--folds', default=4, type=int)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

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

def preprocess(samples, steer_threshold, drop_low_angles):
    samples = np.array(samples, dtype = object)

    print("Number of samples before dropping low steering angles: {}".format(samples.shape[0]))
    index = np.where( np.all(np.abs(samples[:,:,1]) < steer_threshold, axis=1) == True)[0]
    #print(np.abs(samples[:,:,1]) < steer_threshold)
    #print( np.all(np.abs(samples[:,:,1]) < steer_threshold, axis=1) )
    #print(index)
    if drop_low_angles == False:
        rows = [i for i in index if np.random.randint(10) < 9]
    else:
        rows = index
    samples = np.delete(samples, rows, 0)
    print("Removed %s rows with low steering"%(len(rows)))
    print("Number of samples after dropping low steering angles: {}".format(samples.shape[0]))

    return samples

def make_full_dataset(dir, direction, train, steer_threshold, drop_low_angles, steer_correction, window_len, lookahead):
    images = []
    fnames = []
    df = pd.read_csv("Ch2/interpolated.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = "Ch2/{}".format(row[1]['filename'])
        if 'center' in fname:
            fnames.append(fname)
            path = os.path.join(dir, fname)
            item = (path, angle)
            images.append(item)

    lookahead_window = 40

    seq_images = []
    for ix, im in enumerate(images[:-lookahead_window]):
         if ix < window_len-1:
             continue
         fim = [images[ix+lookahead_window], images[ix]]
         if lookahead == True:
             seq_images.append(fim)
         else:
             seq_images.append(images[ix-window_len+1:ix+1])

    return np.array(seq_images), fnames

def make_seq_dataset(dir, direction, train, steer_threshold, drop_low_angles, steer_correction, window_len, lookahead):
    images = []

    df = pd.read_csv("Ch2/interpolated.csv")
    for row in df.iterrows():
        angle = row[1]['angle']
        fname = "Ch2/{}".format(row[1]['filename'])
        if 'center' in fname:
            path = os.path.join(dir, fname)
            item = (path, angle)
            images.append(item)

    lookahead_window = 40

    seq_images = []
    for ix, im in enumerate(images[:-lookahead_window]):
         if ix < window_len-1:
             continue
         fim = [images[ix+lookahead_window], images[ix]]
         #print("a",images[ix+lookahead_window])
         #print("c",images[ix])
         if lookahead == True:
             seq_images.append(fim)
         else:
             seq_images.append(images[ix-window_len+1:ix+1])

    chunk_size = int(len(seq_images)/5)
    train_chunk = int(chunk_size*0.9)
    print(chunk_size, train_chunk)

    train_imgs = []
    val_imgs = []

    for ix in range(0,len(seq_images), chunk_size):
        chunk = seq_images[ix:ix+chunk_size]
        train_imgs.extend(chunk[:train_chunk])
        val_imgs.extend(chunk[train_chunk:])

    #seq_images = preprocess(seq_images, steer_threshold, drop_low_angles)
    return np.array(train_imgs), np.array(val_imgs)

def seq_generator(samples, batch_size, augmentation=False, output='angle', mode='concat'):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # print("Getting next batch")
            batch_samples = samples[offset:offset+batch_size]

            seq_images = []
            y = []
            for bs in batch_samples:
                images = []
                outs = None
                for bi in bs:
                    img_name  = bi[0]
                    steering = np.float32(bi[1])
                    img = cv2.imread(img_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.asarray(img)
                    img = img[:,:,np.newaxis]
                    img = img - 127
                    images.append(img)
                    outs = steering
                #print("Raw images shape: ", np.asarray(images).shape)
                if mode == 'diff':
                    imdiff = [im1-im2 for im1,im2 in zip(images[1:], images[:-1])]
                    #print("Diff images shape: ", np.asarray(imdiff).shape)
                    seq_images.append(np.concatenate(imdiff, axis=2))
                if mode == 'concat':
                    seq_images.append(np.concatenate(images, axis=2)) #images[0]-images[1])
                #print(np.mean(seq_images[-1]))
                y.append(outs)

            X_train = np.array(seq_images)
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

    train_imgs, val_imgs = make_seq_dataset(args.root_dir, args.direction,
                                        True, args.steer_threshold, args.drop_low_angles,
                                        args.steer_correction, args.window_len, args.lookahead)


    print(train_imgs.shape)
    #plt.plot(val_imgs[:,-1,1])
    #plt.show()

    if args.num_samples > 0:
        train_imgs = train_imgs[:args.num_samples]
        val_imgs = val_imgs[:1000]

    train_gen = seq_generator(train_imgs, args.batch_size, args.augmentation, args.output, args.mode)
    val_gen = seq_generator(val_imgs, args.batch_size, False, args.output, args.mode)

    num_outputs = 1 if args.output == 'angle' else 3
    num_inputs = args.window_len if args.mode == 'concat' else args.window_len-1
    model = get_lstm_model(num_outputs, args.l2reg, num_inputs)

    """
    model.load_weights("finalcnn-0.179.h5")

    print(model.predict(x.reshape(-1, 480, 640, 3)).reshape(4,4))

    seq_model =  Sequential()

    return
    """
    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer=adam, metrics=[angle_loss])


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

    pred = []
    real = []
    full_imgs, fnames = make_full_dataset(args.root_dir, args.direction,
                                        True, args.steer_threshold, args.drop_low_angles,
                                        args.steer_correction, args.window_len, args.lookahead)
    full_gen = seq_generator(full_imgs, args.batch_size, args.augmentation, args.output, args.mode)

    print("Full imgs shape: ", full_imgs.shape)
    for ix in range(int(len(full_imgs)/args.batch_size)):
        inp, _real = next(full_gen)
        _pred = model.predict(inp)
        #print(_pred[:,0].shape, _real.shape)
        real.extend(_real)
        pred.extend(_pred[:,0])

    pred = np.array(pred)
    real = np.array(real)
    print("Mean Error: ", np.sqrt(np.mean((pred-real)**2)) )

    plt.figure(figsize=(16,9))
    plt.plot(pred, label='Predicted')
    plt.plot(real, label='Actual')
    plt.legend()
    plt.savefig('pred_baseline.png')
    #plt.show()

    print(len(fnames), pred.shape, real.shape)
    df = pd.DataFrame()
    df['fnames'] = fnames[:len(real)]
    df['angle'] =  real
    df['pred'] = pred
    df.to_csv('results_baseline.csv')

    print(history_object.history['val_loss'])
    print(history_object.history['loss'])

if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
  with K.get_session():
      main(args)
