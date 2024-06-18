import torchvision.transforms as transforms
import random
import numpy as np
from datetime import datetime
from ConfigTypes import NormalizationData, ModelMidSizes, TransformType, Normalizations


rootDir = "/home/kyr/GazeForensicsData/"

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

class RandomResizeTransforms(object):
    def __call__(self, img):
        if random.random() < 0.5:
            size = random.randint(48, 223)
            img = img.resize((size, size))
        return img.resize((224, 224))


def get_standard_transform(nData: NormalizationData):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(177),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=nData.mean, std=nData.std),
        ]
    )


def get_argument_transform(nData: NormalizationData):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (224, 224), scale=(0.7, 0.9), ratio=(0.8, 1.2)
            ),
            # RandomResizeTransforms(),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=nData.mean, std=nData.std),
        ]
    )


def load_lagacy_config(config):
    # Convert the old dict config to the new class config
    new_config = Config()
    # Basic
    new_config.basic.rootDir = config.basic['rootDir']
    new_config.basic.train_DS_name = config.basic['train_DS_name']
    new_config.basic.test_DS_name = config.basic['test_DS_name']
    new_config.basic.normalization = config.basic['normalization']
    new_config.basic.checkpointDir = config.basic['checkpointDir']
    new_config.basic.tryID = config.basic['tryID']
    new_config.basic.evalID = config.basic['tryID']
    # Model
    new_config.model.checkpoint = config.model['checkpoint']
    new_config.model.emb_dim = config.model['emb_dim']
    new_config.model.gaze_backend_path = config.model['gaze_backend_path']
    new_config.model.seed = config.model['seed']
    new_config.model.leaky = config.model['leaky']
    new_config.model.head_num = config.model['head_num']
    new_config.model.dim_per_head = config.model['dim_per_head']
    new_config.model.comp_dim = config.model['comp_dim']
    new_config.model.mid_sizes[ModelMidSizes.gaze_fc] = config.model['mid_sizes']['gaze_fc']
    new_config.model.mid_sizes[ModelMidSizes.MHA_fc] = config.model['mid_sizes']['MHA_fc']
    new_config.model.mid_sizes[ModelMidSizes.MHA_comp] = config.model['mid_sizes']['MHA_comp']
    new_config.model.mid_sizes[ModelMidSizes.last_fc] = config.model['mid_sizes']['last_fc']
    new_config.model.freeze_backend = config.model['freeze_backend']
    # Opti
    new_config.opti.lr = config.opti['lr']
    new_config.opti.weight_decay = config.opti['weight_decay']
    new_config.opti.backend_pct_start = config.opti['backend_pct_start']
    new_config.opti.backend_div_factor = config.opti['backend_div_factor']
    new_config.opti.classifier_pct_start = config.opti['classifier_pct_start']
    new_config.opti.classifier_div_factor = config.opti['classifier_div_factor']
    new_config.opti.enable_grad_clip = config.opti['enable_grad_clip']
    new_config.opti.grad_clip_ref_range = config.opti['grad_clip_ref_range']
    new_config.opti.grad_clip_mul = config.opti['grad_clip_mul']
    # Loss
    new_config.loss.loss_func = config.loss['loss_func']
    new_config.loss.gaze_weight = config.loss['gaze_weight']
    new_config.loss.gaze_gamma = config.loss['gaze_gamma']
    # Prep
    new_config.prep.thread_num = config.prep['thread_num']
    new_config.prep.util_percent = config.prep['util_percent']
    new_config.prep.tempDir = config.prep['tempDir']
    new_config.prep.prep_every_epoch = config.prep['prep_every_epoch']
    new_config.prep.save_dataset = config.prep['save_dataset']
    new_config.prep.transform[TransformType.standard] = config.prep['transform']['standard']
    new_config.prep.transform[TransformType.argument] = config.prep['transform']['argument']
    new_config.prep.argument = config.prep['argument']
    new_config.prep.rand_horizontal_flip = config.prep['rand_horizontal_flip']
    new_config.prep.trainCateDictDir = config.prep['trainCateDictDir']
    # Test
    new_config.test.enable = config.test['enable']
    new_config.test.batch_size = config.test['batch_size']
    new_config.test.num_workers = config.test['num_workers']
    new_config.test.util_percent = config.test['util_percent']
    new_config.test.testClipsDir = config.test['testClipsDir']
    new_config.test.resultDir = config.test['resultDir']
    new_config.test.transform[TransformType.standard] = config.test['transform']['standard']
    new_config.test.verbose = config.test['verbose']
    new_config.test.onlyEvalLastN = config.test['onlyEvalLastN']
    # Train
    new_config.train.enable = config.train['enable']
    new_config.train.num_epochs = config.train['num_epochs']
    new_config.train.batch_size = config.train['batch_size']
    new_config.train.num_workers = config.train['num_workers']
    new_config.train.seed = config.train['seed']
    return new_config


class ParamBasic:  # Fundamental parameters
    def __init__(self):
        self.rootDir = rootDir
        self.train_DS_name = "WDF"
        self.test_DS_name = "WDF"
        self.normalization: NormalizationData = Normalizations.Image_Net
        self.checkpointDir = "Checkpoints/"
        self.tryID = None  # If None, tryID will be set to the current time
        self.evalID = None # If None, evalID will be set to the same as tryID

class ParamModel:  # Parameters that controls model
    def __init__(self):
        # None stands for init a new model, otherwise load the model from the path
        self.checkpoint = None
        self.emb_dim = 512  # Dimension of gaze embedding
        self.gaze_backend_path = "Checkpoints/"  # Will be edited in apply()
        self.seed = 42  # This seed is for initializing the model
        self.leaky = 32  # How many dimensions to expand on gaze embedding
        self.head_num = 8  # Number of heads in multi-head attention
        self.dim_per_head = 64  # Dimension per head in multi-head attention
        self.comp_dim = 32  # Dimension of the compressed feature
        self.mid_sizes = {
            ModelMidSizes.gaze_fc: [],
            ModelMidSizes.MHA_fc: [512],
            ModelMidSizes.MHA_comp: [256, 128],
            ModelMidSizes.last_fc: [256, 128, 64],
        }
        # Freeze backend and use gaze as backend, leaky will still work if not zero.
        self.freeze_backend = False


class ParamOpti:  # Parameters that controls optimizing, loss, scheduling.
    def __init__(self):
        self.lr = 7e-5
        self.weight_decay = 1e-2

        self.backend_pct_start = 0.3
        self.backend_div_factor = 1e3
        self.classifier_pct_start = 0.3
        self.classifier_div_factor = 1e3

        self.enable_grad_clip = False
        self.grad_clip_ref_range = 16  # The range of reference value for gradient clipping
        self.grad_clip_mul = 1.5


class ParamLoss:  # Parameters that controls loss
    def __init__(self):
        self.loss_func = 'standard'
        # self.bonus_weight = 0
        # self.FN_w = 1 # False negative weight in custom loss
        # self.FN_bound = 0.6 # False negative boundary in custom loss
        self.gaze_weight = 4.0
        #   'gaze_weight' decides whether to use gaze info or not
        #   because final_loss = output_BCE_loss + gaze_weight * gaze_MSE_loss
        #   if gaze_weight is 0, the model will not use gaze info
        self.gaze_gamma = None


class ParamPrep:  # Parameters that controls preprocess
    def __init__(self):
        # Note that different thread_num will lead to different result in dataset clipping
        self.thread_num = 4
        self.util_percent = 0.2
        #   utilization percentage of the dataset
        #   DO NOT use 1.0, this will help to avoid overfitting
        self.tempDir = rootDir + "temp/"
        # 'tempDir' is the directory to store clipped train dataset
        self.prep_every_epoch = True
        #   True or False
        #   Due to the randomness of the dataset clipping process,
        #   preprocess the dataset every epoch may alleviate overfitting
        self.save_dataset = True
        # This is useful when trying to find the best hyperparameters
        self.transform = {
            TransformType.standard: None,  # Auto complete in apply()
            TransformType.argument: None,  # Auto complete in apply()
        }
        self.argument = False  # Use argument_transform or not
        self.rand_horizontal_flip = True  # Use random horizontal flip or not
        self.trainCateDictDir = None  # Auto complete in apply()


class ParamTest:  # Parameters that controls test process
    def __init__(self):
        self.enable: bool = True
        self.batch_size: int = 16
        self.num_workers: int = 4
        self.util_percent: float = 1.0  # utilization percentage of the dataset
        self.testClipsDir: str = None  # Auto complete in apply()
        self.resultDir: str = "Results/"
        self.transform: dict = {
            TransformType.standard: None
        }  # Auto complete in apply()
        self.verbose: bool = True
        self.onlyEvalLastN: int = None  # If not None, only evaluate the last N checkpoints


class ParamTrain:  # Parameters that controls train process
    def __init__(self):
        self.enable: bool = True
        self.num_epochs: int = 25
        self.batch_size: int = 32
        self.num_workers: int = 4
        self.seed: int = 42
        self.smooth_label_alpha: float = 0
        self.smooth_label_beta: float = 0
        #   Smoothed label = abs(label - beta * uniform_distribution - alpha)
        self.verbose: bool = True


class Config:

    def __init__(self):
        self.basic: ParamBasic = ParamBasic()
        self.model: ParamModel = ParamModel()
        self.opti: ParamOpti = ParamOpti()
        self.loss: ParamLoss = ParamLoss()
        self.prep: ParamPrep = ParamPrep()
        self.test: ParamTest = ParamTest()
        self.train: ParamTrain = ParamTrain()

    def auto_set_tryID(self):
        self.basic.tryID = datetime.now().strftime("%y%m%d_%H%M%S")

    def apply(self):
        if self.basic.evalID is None:
            self.basic.evalID = self.basic.tryID

        if self.loss.gaze_gamma is None:
            self.loss.gaze_gamma = 0.01 ** (1 / self.train.num_epochs)

        self.opti.lr = self.opti.lr / 32 * self.train.batch_size

        self.test.testClipsDir = (
            rootDir + "clipped_videos/" + self.basic.test_DS_name + "_clip/"
        )
        self.prep.trainCateDictDir = (
            rootDir
            + "clip_info/"
            + self.basic.train_DS_name
            + "_vid_category_dict.json"
        )

        self.prep.transform[TransformType.standard] = get_standard_transform(
            self.basic.normalization
        )

        self.prep.transform[TransformType.argument] = get_argument_transform(
            self.basic.normalization
        )
        self.test.transform[TransformType.standard] = get_standard_transform(
            self.basic.normalization
        )

        self.model.gaze_backend_path += (
            "gazeBackend_" + str(self.model.emb_dim) + ".pkl"
        )

        assert not (self.model.freeze_backend and self.model.finetune_backend)
