import torchvision.transforms as transforms



train_DS_name = 'CDF'
test_DS_name = 'CDF'
rootDir = '/home/kyr/GazeForensicsData/'

standard_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



'''
Serialized parameters start from here
'''
basic_param = { # Fundamental parameters
    'rootDir': rootDir,
    'train_DS_name': train_DS_name,
    'test_DS_name': test_DS_name,
    'checkpointDir': rootDir + 'checkpoints/',
    'tryID': 'temp',
    # 'tryID' specifies the folder name in checkpoint folder
    #  - if tryID is None, the model will not save checkpoints
    #  - if tryID is 'temp', the model will overwrite the temp folder
}
model_param = { # Parameters that controls model
    'leaky': 0,
    'model_path': None, # None stands for init a new model, otherwise load the model from the path
    'arch': 'resnet18',
}
opti_param = { # Parameters that controls optimizing, loss, scheduling.
    'lr': 1e-5,
    'weight_decay': 1e-5,
    'step_size': 1,
    'gamma': 0.99,
    'gaze_loss_weight': 1,
    # 'gaze_loss_weight' decides whether to use gaze info or not
    # because final_loss = output_BCE_loss + gaze_loss_weight * gaze_MSE_loss
    #  - if gaze_loss_weight is 0, the model will not use gaze info
}
prep_param = { # Parameters that controls preprocess
    'thread_num': 4,
    'utilization_percentage': 0.1, # utilization percentage of the dataset
    'tempDir': rootDir + 'temp/',
    'prep_every_epoch': None, # None stands for let condition decide, True and False stand for enforcement
    'transform': standard_transform,
    'trainCateDictDir': rootDir + 'clip_info/' + train_DS_name + '_vid_category_dict.json',
    'testClipsDir': rootDir + 'clipped_videos/' + test_DS_name + '_clip/',
}
test_param = { # Parameters that controls test process
    'batch_size': 28,
    'num_workers': 4,
}
train_param = { # Parameters that controls train process
    'num_epochs': 10,
    'batch_size': 6,
    'num_workers': 4,
}