import torchvision.transforms as transforms
import random
import numpy as np
import json
from datetime import datetime



rootDir = '/home/kyr/GazeForensicsData/'

normalization_mean = {
    'Image-Net':    [0.485, 0.456, 0.406],
    'WDF':          [0.428, 0.338, 0.301],
}
normalization_std = {
    'Image-Net':    [0.229, 0.224, 0.225],
    'WDF':          [0.241, 0.212, 0.215],
}


class RandomResizeTransforms(object):
    def __call__(self, img):
        if random.random() < 0.5:
            size = random.randint(48, 223)
            img = img.resize((size, size))
        return img.resize((224, 224))


def get_standard_transform(mean, std):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_argument_transform(mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.3)),
        # RandomResizeTransforms(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class Config:

    def __init__(self):

        params_basic = { # Fundamental parameters
            'rootDir': rootDir,
            'train_DS_name': 'WDF',
            'test_DS_name': 'WDF',
            'normalization': 'Image-Net',
            'checkpointDir': 'Checkpoints/',
            'tryID': None, # If None, tryID will be set to the current time
        }
        params_model = { # Parameters that controls model
            'checkpoint': None, # None stands for init a new model, otherwise load the model from the path
            'emb_dim': 512, # Dimension of gaze embedding
            'gaze_backend_path': 'Checkpoints/', # Will be edited in apply()
            'seed': 42, # This seed is for initializing the model
            'leaky': 32, # How many dimensions to expand on gaze embedding
            'head_num': 8, # Number of heads in multi-head attention
            'dim_per_head': 64, # Dimension per head in multi-head attention
            'comp_dim': 32, # Dimension of the compressed feature
            'mid_sizes': {
                'gaze_fc': [],
                'MHA_fc': [512],
                'MHA_comp': [256, 128],
                'last_fc': [256, 128, 64],
            },
            'freeze_backend': False, # Freeze backend and use gaze as backend, leaky will still work if not zero.
        }
        params_opti = { # Parameters that controls optimizing, loss, scheduling.
            'lr': 7e-5, # lr = lr / 32 * batch_size, this is auto set in apply() function
            'weight_decay': 1e-2,
            
            'backend_pct_start': 0.3,
            'backend_div_factor': 1e3,
            'classifier_pct_start': 0.3,
            'classifier_div_factor': 1e3,

            'enable_grad_clip': False,
            'grad_clip_ref_range': 16, # The range of reference value for gradient clipping
            'grad_clip_mul': 1.5,
        }
        params_loss = { # Parameters that controls loss
            'loss_func': 'standard',
            # 'bonus_weight': 0,
            # 'FN_w': 1, # False negative weight in custom loss
            # 'FN_bound': 0.6, # False negative boundary in custom loss
            'gaze_weight': 4.0,
            #   'gaze_weight' decides whether to use gaze info or not
            #   because final_loss = output_BCE_loss + gaze_weight * gaze_MSE_loss
            #   if gaze_weight is 0, the model will not use gaze info
            'gaze_gamma': None, # If None, will be auto set in apply() function
        }
        params_prep = { # Parameters that controls preprocess
            'thread_num': 4, # Note that different thread_num will lead to different result in dataset clipping
            'util_percent': 0.2,
            #   utilization percentage of the dataset
            #   DO NOT use 1.0, this will help to avoid overfitting
            'tempDir': rootDir + 'temp/', # 'tempDir' is the directory to store clipped train dataset
            'prep_every_epoch': True,
            #   True or False
            #   Due to the randomness of the dataset clipping process,
            #   preprocess the dataset every epoch may alleviate overfitting
            'save_dataset': True, # True or False, this is useful when trying to find the best hyperparameters
            'transform': {
                'standard': get_standard_transform(
                    normalization_mean['Image-Net'],
                    normalization_std['Image-Net']
                ),
                'argument': None, # Auto complete in apply()
            },
            'argument': False, # Use argument_transform or not
            'rand_horizontal_flip': True, # Use random horizontal flip or not
            'trainCateDictDir': None, # Auto complete in apply()
        }
        params_test = { # Parameters that controls test process
            'enable': True, # True or False
            'batch_size': 16,
            'num_workers': 4,
            'util_percent': 1.0, # utilization percentage of the dataset
            'testClipsDir': None, # Auto complete in apply()
            'resultDir': 'Results/',
            'transform': {'standard': None}, # Auto complete in apply()
            'verbose': True,
            'onlyEvalLastN': None, # If not None, only evaluate the last N checkpoints
        }
        params_train = { # Parameters that controls train process
            'enable': True, # True or False
            'num_epochs': 25,
            'batch_size': 32,
            'num_workers': 4,
            'seed': 42, # This seed is for the random shuffle of the dataset in training
            'smooth_label_alpha': 0,
            'smooth_label_beta': 0,
            #   Smoothed label = abs(label - beta * uniform_distribution - alpha)
            'verbose': True,
        }


        self.basic = params_basic
        self.model = params_model
        self.opti = params_opti
        self.prep = params_prep
        self.test = params_test
        self.train = params_train
        self.loss = params_loss
    


    def auto_set_tryID(self):
        self.basic['tryID'] = datetime.now().strftime("%y%m%d_%H%M%S")

    

    def apply(self):
        if self.loss['gaze_gamma'] is None:
            self.loss['gaze_gamma'] = 0.01 ** (1 / self.train['num_epochs'])
        
        self.opti['lr'] = self.opti['lr'] / 32 * self.train['batch_size']

        self.test['testClipsDir'] = rootDir + 'clipped_videos/' + self.basic['test_DS_name'] + '_clip/'
        self.prep['trainCateDictDir'] = rootDir + 'clip_info/' + self.basic['train_DS_name'] + '_vid_category_dict.json'

        self.prep['transform']['argument'] = get_argument_transform(
            normalization_mean[self.basic['normalization']],
            normalization_std[self.basic['normalization']]
        )
        self.test['transform']['standard'] = get_standard_transform(
            normalization_mean[self.basic['normalization']],
            normalization_std[self.basic['normalization']]
        )

        self.model['gaze_backend_path'] += 'gazeBackend_' + str(self.model['emb_dim']) + '.pkl'



class ParameterTable:
    def __init__(self, num_epochs, saved_param_sets=[]):
        def init_param_ranges():
            seed_table = [2**i for i in range(13)] + [0, 2022, 2023, 2024, 65535]
            # Model
            self.model_seed = seed_table[:]
            self.leaky = [i for i in np.arange(0, 257, 32)]
            self.head_dim = [4, 6, 8]
            self.dim_per_head = [32, 48, 64]
            self.comp_dim = [32, 48, 64, 80, 96]
            self.gaze_fc_layer_num = [0, 1, 2]
            self.MHA_fc_layer_num = [0, 1, 2]
            self.MHA_comp_layer_num = [1, 2]
            self.last_fc_layer_num = [1, 2, 3]
            # Opti
            self.lr = [i for i in np.arange(1e-5, 2.5e-5, 0.2e-5)]
            self.weight_decay = [1e-6, 1e-5, 1e-4, 1e-3]
            self.gamma = [i for i in np.arange(0.93, 0.95, 0.01)]
            # Loss
            self.gaze_weight = [i for i in np.arange(1, 12, 0.5)]
            self.gaze_gamma = [i for i in np.arange(0.8, 0.95, 0.01)]
            # Train
            self.train_seed = seed_table[:]
            self.smooth_label_alpha = [0.05, 0.1, 0.15]
        
        self.num_epochs = num_epochs
        self.saved_param_sets = saved_param_sets
        self.param_set = None
        init_param_ranges()
    
    def ChooseParam(self):
        def choose_param(num_epochs):
            model_seed = np.random.choice(self.model_seed, 1)[0]
            leaky = np.random.choice(self.leaky, 1)[0]
            head_dim = np.random.choice(self.head_dim, 1)[0]
            dim_per_head = np.random.choice(self.dim_per_head, 1)[0]
            comp_dim = np.random.choice(self.comp_dim, 1)[0]
            gaze_fc_layer_num = np.random.choice(self.gaze_fc_layer_num, 1)[0]
            MHA_fc_layer_num = np.random.choice(self.MHA_fc_layer_num, 1)[0]
            MHA_comp_layer_num = np.random.choice(self.MHA_comp_layer_num, 1)[0]
            last_fc_layer_num = np.random.choice(self.last_fc_layer_num, 1)[0]
            lr = np.random.choice(self.lr, 1)[0]
            weight_decay = np.random.choice(self.weight_decay, 1)[0]
            gamma = np.random.choice(self.gamma, 1)[0] ** (10 / num_epochs)
            gaze_weight = np.random.choice(self.gaze_weight, 1)[0]
            gaze_gamma = np.random.choice(self.gaze_gamma, 1)[0] ** (10 / num_epochs)
            train_seed = np.random.choice(self.train_seed, 1)[0]
            smooth_label_alpha = np.random.choice(self.smooth_label_alpha, 1)[0]
            param_set = {
                'model_seed': model_seed,
                'leaky': leaky,
                'head_dim': head_dim,
                'dim_per_head': dim_per_head,
                'comp_dim': comp_dim,
                'gaze_fc_layer_num': gaze_fc_layer_num,
                'MHA_fc_layer_num': MHA_fc_layer_num,
                'MHA_comp_layer_num': MHA_comp_layer_num,
                'last_fc_layer_num': last_fc_layer_num,
                'lr': lr,
                'weight_decay': weight_decay,
                'gamma': gamma,
                'gaze_weight': gaze_weight,
                'gaze_gamma': gaze_gamma,
                'train_seed': train_seed,
                'smooth_label_alpha': smooth_label_alpha,
            }
            return param_set
        
        while True:
            param_set = choose_param(self.num_epochs)
            if param_set not in self.saved_param_sets:
                break
        
        # Convert all numpy types to python types
        for key in param_set:
            param_set[key] = param_set[key].item()

        self.param_set = param_set
        # print('>> Current parameter set:')
        # display(param_set)
    

    def SaveParamSet(self, tryID):
        with open('Results/' + tryID + '_paramSet.json', 'w') as f:
            json.dump(self.param_set, f)
    

    def LoadParamSet(self, tryID):
        with open('Results/' + tryID + '_paramSet.json', 'r') as f:
            self.param_set = json.load(f)
    

    def ApplyToConfig(self, config:Config):
        # Model
        config.model['seed'] = self.param_set['model_seed']
        config.model['leaky'] = self.param_set['leaky']
        config.model['head_dim'] = self.param_set['head_dim']
        config.model['dim_per_head'] = self.param_set['dim_per_head']
        config.model['comp_dim'] = self.param_set['comp_dim']

        MHA_dim = self.param_set['head_dim'] * self.param_set['dim_per_head']
        last_fc_input_size = self.param_set['comp_dim'] * 14

        config.model['mid_sizes']['gaze_fc'] = [512] * self.param_set['gaze_fc_layer_num']
        config.model['mid_sizes']['MHA_fc'] = [MHA_dim] * self.param_set['MHA_fc_layer_num']
        if self.param_set['MHA_comp_layer_num'] == 1:
            config.model['mid_sizes']['MHA_comp'] = [
                int((MHA_dim - self.param_set['comp_dim']) * 0.4) + self.param_set['comp_dim']
            ]
        elif self.param_set['MHA_comp_layer_num'] == 2:
            config.model['mid_sizes']['MHA_comp'] = [
                int((MHA_dim - self.param_set['comp_dim']) * 0.5) + self.param_set['comp_dim'],
                int((MHA_dim - self.param_set['comp_dim']) * 0.2) + self.param_set['comp_dim'],
            ]
        if self.param_set['last_fc_layer_num'] == 1:
            config.model['mid_sizes']['last_fc'] = [last_fc_input_size // 2]
        elif self.param_set['last_fc_layer_num'] == 2:
            config.model['mid_sizes']['last_fc'] = [last_fc_input_size // 2, last_fc_input_size // 4]
        elif self.param_set['last_fc_layer_num'] == 3:
            config.model['mid_sizes']['last_fc'] = [
                last_fc_input_size // 2,
                last_fc_input_size // 4,
                last_fc_input_size // 8
            ]

        # Opti
        config.opti['lr'] = self.param_set['lr']
        config.opti['weight_decay'] = self.param_set['weight_decay']
        config.opti['gamma'] = self.param_set['gamma']

        # Loss
        config.loss['gaze_weight'] = self.param_set['gaze_weight']
        config.loss['gaze_gamma'] = self.param_set['gaze_gamma']

        # Train
        config.train['seed'] = self.param_set['train_seed']
        config.train['smooth_label_alpha'] = self.param_set['smooth_label_alpha']
    

    def GetListParamSet(self):
        paramList = []
        for key in self.param_set.keys():
            paramList.append(self.param_set[key])
        return paramList
    

    def SetParamSetByList(self, param_set_list):
        if self.param_set is None:
            self.ChooseParam()
        for key, value in zip(self.param_set.keys(), param_set_list):
            self.param_set[key] = value
    

    def Index2Key(self, index):
        key = list(self.param_set.keys())[index]
        return key