import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import gc
import time


TRAIN_DIR =r'C:\Users\Divij\Desktop\everythingPython\deepLearning\DogsNCats\train'
TEST_DIR =r'C:\Users\Divij\Desktop\everythingPython\deepLearning\DogsNCats\test'

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


def create_train_data():
    
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        start = time.time()
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        elapsed_time_fl = (time.time() - start)
        print("\n\n\n\n\n  elapsed time            :"+str(elapsed_time_fl)+"\n\n\n\n\n");
   
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    gc.collect()
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        start = time.time()
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        elapsed_time_fl = (time.time() - start)
        print("\n\n\n\n\n  elapsed time            :"+str(elapsed_time_fl)+"\n\n\n\n\n");
   
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

test_data = process_test_data()
