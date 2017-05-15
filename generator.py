import glob, os

import numpy as np
import pandas as pd
import scipy.misc

class DataReader(object):
    
    def __init__(self, data_dir="./data", file_ext=".jpg", sequential=False):
        self.data_dir = data_dir
        self.sequential = sequential
        self.num_train = 0
        self.num_val = 0
        self.prepare_training()
        self.prepare_val()
        
        self.train_index = 0
        self.val_index = 0
        
    def prepare_training(self, direction='center'):
        
        if direction == 'center':
            files = [f for f in glob.glob("data/train_center*.csv")]
        elif direction == 'left':
            files = [f for f in glob.glob("data/train_left*.csv")]
            print files
        elif direction == 'right':
            files = [f for f in glob.glob("data/train_right*.csv")]
        else:
            pass
        
        self.train_paths = []
        self.train_y = []
        for f in files:
            df = pd.read_csv(f, dtype={'angle': np.double,'torque': np.double, 'speed': np.double})
            for row in df.iterrows():
                fn = "Ch2/{}".format(row[1]['filename'])
                self.train_paths.append(fn)
                self.train_y.append(row[1]['angle'])
                self.num_train += 1
    
    def prepare_val(self, direction='center'):
        
        if direction == 'center':
            files = [f for f in glob.glob("data/val_center*.csv")]
        elif direction == 'left':
            files = [f for f in glob.glob("data/val_left*.csv")]
            print files
        elif direction == 'right':
            files = [f for f in glob.glob("data/val_right*.csv")]
        else:
            pass
        
        self.val_paths = []
        self.val_y = []
        for f in files:
            df = pd.read_csv(f, dtype={'angle': np.double,'torque': np.double, 'speed': np.double})
            for row in df.iterrows():
                fn = "Ch2/{}".format(row[1]['filename'])
                self.val_paths.append(fn)
                self.val_y.append(row[1]['angle'])
                self.num_val += 1
                
    def load_train_minibatch(self, batch_size):
        x = []
        y = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_paths[(self.train_index + i) % self.num_train])
            x.append(image / 255.0)
            y.append([self.train_y[(self.train_index + i) % self.num_train]])
        self.train_index += batch_size
        return x,y
    
    def load_val_minibatch(self, batch_size):
        x = []
        y = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.val_paths[(self.val_index + i) % self.num_val])
            x.append(image / 255.0)
            y.append([self.val_y[(self.val_index + i) % self.num_val]])
        self.val_index += batch_size
        return x,y
