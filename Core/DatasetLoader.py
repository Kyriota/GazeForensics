from Utils.FileOperation import fileWalk

import torch
import torch.utils.data as TDM # torch data module

import cv2
from PIL import Image



class DatasetLoader(TDM.Dataset):
    def __init__(self, data_dir_list, transform):
        self.dataset = data_dir_list
        self.transform = transform


    def __getitem__(self, index):
        sign, vid_path = self.dataset[index]
        inputTensor = torch.FloatTensor(14, 3, 224, 224)
        vidcap = cv2.VideoCapture(vid_path)
        # assert vidcap.isOpened(), '\n>>> ERROR: Failed to open the video: ' + vid_path + '\n'
        # assert vidcap.get(cv2.CAP_PROP_FRAME_COUNT) == 14, '\n>>> ERROR: The video is not 14 frames long: ' + vid_path + '\n'
        for i in range(14):
            _, image = vidcap.read()
            inputTensor[i] = self.transform(Image.fromarray(image))
        vidcap.release()

        return [sign, inputTensor[:, [2, 1, 0], :, :]]
    

    def __len__(self):
        return len(self.dataset)