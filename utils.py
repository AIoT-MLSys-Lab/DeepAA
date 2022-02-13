import os
import logging
import numpy as np
import matplotlib
# configure backend here
matplotlib.use('Agg')
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import math
import sys
from data_generator import CIFAR_MEANS, CIFAR_STDS

gfile = tf.io.gfile

class Logger(object):
  """Prints to both STDOUT and a file."""

  def __init__(self, filepath):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filepath, 'a+')

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.terminal.flush()
    self.log.flush()

class CTLEarlyStopping:
    def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               mode='auto',
               ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stop_training = False
        self.improvement = False

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
            
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    
    def check_progress(self, current):
        if self.monitor_op(current - self.min_delta, self.best):
            print(f"{self.monitor} improved from {self.best:.4f} to {current:.4f}.", end=" ")
            self.best = current
            self.wait = 0
            self.improvement = True
        else:
            self.wait += 1
            self.improvement = False
            print(f"{self.monitor} didn't improve")
            if self.wait >= self.patience:
                print("Early stopping")
                self.stop_training = True
                
        return self.improvement, self.stop_training
    
    
##########################################################################################

    
class CTLHistory:
    def __init__(self,
                 filename=None,
                 save_dir='plots'):
        
        self.history = {'train_loss':[], 
                        "train_acc":[], 
                        "val_loss":[], 
                        "val_acc":[],
                        "lr":[],
                        "wd":[]}
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        try:
            filename = 'history_cuda.png'
        except:
            filename = 'history.png' if filename is None else filename

        self.plot_name = os.path.join(self.save_dir, filename)
    
   
  
    def update(self, train_stats, val_stats, record_lr_wd):
        train_loss, train_acc = train_stats
        val_loss, val_acc = val_stats
        lr_history, wd_history = record_lr_wd
        
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(np.round(train_acc*100))
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(np.round(val_acc*100))
        self.history['lr'].extend(lr_history)
        self.history['wd'].extend(wd_history)

        
    def plot_and_save(self, initial_epoch=0):
        train_loss = self.history['train_loss']
        train_acc = self.history['train_acc']
        val_loss = self.history['val_loss']
        val_acc = self.history['val_acc']
        
        epochs = [(i+initial_epoch) for i in range(len(train_loss))]
        
        f, ax = plt.subplots(3, 1, figsize=(15,8))
        ax[0].plot(epochs, train_loss)
        ax[0].plot(epochs, val_loss)
        ax[0].set_title('loss progression')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('loss values')
        ax[0].legend(['train', 'test'])
        
        ax[1].plot(epochs, train_acc)
        ax[1].plot(epochs, val_acc)
        ax[1].set_title('accuracy progression')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(['train', 'test'])

        steps = len(self.history['lr'])
        bs = steps/len(train_loss)
        ax[2].plot([s/bs for s in range(steps)], self.history['lr'])
        ax[2].plot([s/bs for s in range(steps)], self.history['wd'])
        ax[2].set_title('learning rate and weight decay')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('lr and wd')
        ax[2].legend(['lr', 'wd'])

        plt.savefig(self.plot_name)
        plt.close()

def repeat(x, n, axis):
    if isinstance(x, np.ndarray):
        return np.repeat(x, n, axis=axis)
    elif isinstance(x, list):
        return repeat_list(x, n, axis)
    else:
        raise Exception('Unsupport data type {}'.format(type(x)))

def repeat_list(x, n, axis):
    assert isinstance(x, list), 'Can only consume list type'
    if axis == 0:
        x_new = sum([[x_] * n for x_ in x], [])
    elif axis > 1:
        x_new = [repeat(x_, n, axis=axis - 1) for x_ in x]
    else:
        raise Exception
    return x_new

def tile(x):
    return None