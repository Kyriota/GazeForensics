import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from batch_face import RetinaFace
from _backend import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--input',dest='input_path', help='Path of input video.',
    )
    parser.add_argument(
        '--output',dest='output_path', help='Path of output video.',
    )
    parser.add_argument(
        '--use_detector',dest='use_detector', help='Use face detector or not.', type=bool, default=True
    )
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--blur',dest='blur',help='blur the video or not',
        default=False, type=bool)
    parser.add_argument(
        '--emb_dim',dest='emb_dim',help='Dimension of embedding vector',
        default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    input_path = args.input_path
    output_path = args.output_path
    use_detector = args.use_detector
    snapshot_path = args.snapshot
    blur = args.blur
    emb_dim = args.emb_dim
   
    if blur:
        print('Video will be blurred.')
        transformations = transforms.Compose([
            transforms.Resize(32),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    model=L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 90, eval_mode=True, emb_dim=emb_dim)
    saved_state_dict = torch.load('models/L2CSNet_gaze360.pkl')
    # Only load "fc_yaw_gaze.weight", "fc_yaw_gaze.bias", "fc_pitch_gaze.weight", "fc_pitch_gaze.bias"
    to_be_loaded = ['fc_yaw_gaze.weight', 'fc_yaw_gaze.bias', 'fc_pitch_gaze.weight', 'fc_pitch_gaze.bias']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in saved_state_dict.items() if k in to_be_loaded}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    saved_state_dict = torch.load(snapshot_path)
    print(model.load_state_dict(saved_state_dict, strict=False))

    model.cuda()
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    # Load input video
    video = cv2.VideoCapture(input_path)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()

    with torch.no_grad():

        for i in range(len(frames)):
            
            image = frames[i]

            faces = detector(image)
            if faces is not None: 
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = image[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda()
                    img = img.unsqueeze(0) 

                    # Gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

                    draw_gaze(x_min, y_min, bbox_width, bbox_height, image, (pitch_predicted, yaw_predicted), color=(0,0,255))
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            
            frames[i] = image
    
    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()