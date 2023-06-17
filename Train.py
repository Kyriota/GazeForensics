import sys

if 'Models/' not in sys.path:
    sys.path.append('Models/')
from model import GazeForensics

if 'Utils/' not in sys.path:
    sys.path.append('Utils/')
from FileOperation import *
from DatasetClipping import ClipTrainDatasetParallel
from Visualiztion import ProgressBar

import torch
import torch.nn as nn
import torch.utils.data as TDM # torch data module

import cv2
from PIL import Image
import threading
from threading import Lock
import time



def InitModel(hyp): # hyp: hyper parameters
    model = GazeForensics(leaky=hyp['leaky'], save_mid_layer=hyp['save_mid_layer'], save_last_layer=hyp['save_last_layer'])
    model = nn.DataParallel(model).cuda()
    return model



class DatasetLoader(TDM.Dataset):
    def __init__(self, clippedDataPath, transform):
        self.transform = transform
        self.dataset = [clippedDataPath + i for i in fileWalk(clippedDataPath)]
        self.dataset = [[0 if 'fake' in path else 1, path] for path in self.dataset]

    def __getitem__(self, index):
        sign, vid_path = self.dataset[index]
        inputTensor = torch.FloatTensor(14, 3, 224, 224)
        vidcap = cv2.VideoCapture(vid_path)
        assert vidcap.isOpened(), '\n>>> ERROR: Failed to open the video: ' + vid_path + '\n'
        assert vidcap.get(cv2.CAP_PROP_FRAME_COUNT) == 14, '\n>>> ERROR: The video is not 14 frames long: ' + vid_path + '\n'
        for i in range(14):
            _, image = vidcap.read()
            inputTensor[i,...] = self.transform(Image.fromarray(image))
        vidcap.release()
        inputTensor = inputTensor[:, [2, 1, 0], :, :]

        return [sign, inputTensor]

    def __len__(self):
        return len(self.dataset)



def AddLock(target, lock):
    def wrapper(*args, **kwargs):
        lock.acquire()
        ret = target(*args, **kwargs)
        lock.release()
        return ret
    return wrapper



class TrainManager:
    def __init__(self, nop, hyp):
        self.nop = nop # normal parameters dict
        self.hyp = hyp # hyper parameters dict
        self.preprocess_lock = Lock()
        self.locked_clipper = AddLock(ClipTrainDatasetParallel, self.preprocess_lock)
    
    def train(self):

        print('Function outer train() is called.')

        def get_preprocess_thread(index):
            return threading.Thread(
                target=self.locked_clipper,
                kwargs={
                    'json_path': self.nop['rootDir'] + self.nop['trainCateDictDir'],
                    'output_path': self.nop['rootDir'] + self.nop['tempDir'] + str(index % 2) + '/',
                    'utilization_percentage': self.nop['utilization_percentage'],
                }
            )

        def train(epoch):
            print('Function inner train() is called.')
            for i in range(10):
                print('Training process', epoch, str(i+1) + '/10')
                time.sleep(0.2)
            print('Training process', epoch, 'finished.')
        
        # Preprocess the training dataset
        preprocess_thread = get_preprocess_thread(0)
        preprocess_thread.start()

        # Initialize the model
        model = InitModel(self.hyp)

        # Init optimizer, loss function, etc.
        pass

        # Wait for the first preprocessing to finish
        self.preprocess_lock.acquire()
        self.preprocess_lock.release()

        # Start training
        for epoch in range(self.nop['num_epochs']):
            #if self.nop['utilization_percentage'] == 1.0:
            if epoch != self.nop['num_epochs'] - 1:
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(epoch+1)
                preprocess_thread.start()

            # Train the model
            train(epoch)

            # Wait for the preprocessing to finish
            self.preprocess_lock.acquire()
            self.preprocess_lock.release()

            print('Epoch', epoch, 'finished.')