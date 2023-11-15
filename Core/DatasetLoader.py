from Utils.Visualization import NormalizeImage

import torch
import torch.utils.data as TDM # torch data module
import torchvision.transforms as transforms

import cv2
from PIL import Image
import random
import numpy as np



class DatasetLoader(TDM.Dataset):
    def __init__(
            self,
            data_dir_list,
            transform,
            use_argument=False,
            return_path=False,
            rand_horizontal_flip=False
        ):
        self.dataset = data_dir_list
        self.transform = transform
        self.return_path = return_path
        self.use_argument = use_argument
        self.rand_horizontal_flip = rand_horizontal_flip
        random.seed(0)

        if rand_horizontal_flip:
            self.transform_HF = transforms.RandomHorizontalFlip(p=1.0)
    

    def get_normalization(self):
        # return mean and std of dataset
        channels_sum, channels_sqrd_sum = [], []
        pixel_num = 0
        for i in range(len(self.dataset)):
            _, vid_path = self.dataset[i]
            vidcap = cv2.VideoCapture(vid_path)
            temp_channels_sum, temp_channels_sqrd_sum = [], []
            for _ in range(14):
                _, image = vidcap.read()
                image = (image[:, :, ::-1] / 255.0).astype(np.float64) # BGR to RGB
                # Calculate channel sum and channel squared sum
                temp_channels_sum.append(np.sum(image, axis=(0, 1)))
                temp_channels_sqrd_sum.append(np.sum(np.square(image), axis=(0, 1)))
                pixel_num += image.shape[0] * image.shape[1]
            channels_sum.extend(temp_channels_sum)
            channels_sqrd_sum.extend(temp_channels_sqrd_sum)
            vidcap.release()
            # Print progress
            if i % 100 == 0 or i == len(self.dataset) - 1:
                print('Progress: {}/{}'.format(i+1, len(self.dataset)), end='\r' if i != len(self.dataset) - 1 else '\n')
        channels_sum = np.array(channels_sum).sum(axis=0)
        channels_sqrd_sum = np.array(channels_sqrd_sum).sum(axis=0)
        mean = channels_sum / pixel_num
        std = np.sqrt(channels_sqrd_sum / pixel_num - np.square(mean))
        return mean, std


    def __getitem__(self, index):
        sign, vid_path = self.dataset[index]
        vidcap = cv2.VideoCapture(vid_path)

        HF_flag = np.random.random() > 0.5

        if self.use_argument:
            inputTensor = {
                'standard': torch.FloatTensor(14, 3, 224, 224),
                'argument': torch.FloatTensor(14, 3, 224, 224),
            }
            for i in range(14):
                _, image = vidcap.read()
                image = Image.fromarray(image[:, :, ::-1])
                if self.rand_horizontal_flip and HF_flag:
                    image = self.transform_HF(image)
                inputTensor['standard'][i] = self.transform['standard'](image)
                inputTensor['argument'][i] = self.transform['argument'](image)

        else:
            inputTensor = {'standard': torch.FloatTensor(14, 3, 224, 224)}
            for i in range(14):
                _, image = vidcap.read()
                image = Image.fromarray(image[:, :, ::-1])
                if self.rand_horizontal_flip and HF_flag:
                    image = self.transform_HF(image)
                inputTensor['standard'][i] = self.transform['standard'](image)

        vidcap.release()

        # # Debug code: Save images using PIL.Image as 'test_{i}.jpg'
        # for i in range(14):
        #     temp_file_name = 'Debug/' + vid_path.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
        #     Image.fromarray(
        #         NormalizeImage(inputTensor[
        #             'argument' if self.use_argument else 'standard'
        #         ][i].numpy().transpose(1, 2, 0))
        #     ).save(temp_file_name)

        if self.return_path:
            return [sign, inputTensor, vid_path]
        return [sign, inputTensor]
    

    def __len__(self):
        return len(self.dataset)