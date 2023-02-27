from pathlib import Path
from enum import Enum

#   Path configuration
class PathConfig:
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent

class DataPathConfig(PathConfig):
    def __init__(self):
            
        super(DataPathConfig, self).__init__()
        self.DATA_DIR = "data"
        self.DATASET     = ["Flickr8K", "Pascal-VOC-2012"]

        ################### Image Captioning ##############
        self.CAPTIONS_PATH = self.BASE_DIR / self.DATA_DIR / self.DATASET[0] / "captions"
        self.IMG_ID_PATH = self.BASE_DIR / self.DATA_DIR / self.DATASET[0] / "images_id"
        self.IMG_CAPTION = self.BASE_DIR / self.DATA_DIR / self.DATASET[0] / "images"

        self.VOC2012_TEST = self.BASE_DIR / self.DATA_DIR / self.DATASET[1] / "VOC2012_test"
        self.VOC2012_TRAIN = self.BASE_DIR / self.DATA_DIR / self.DATASET[1] / "VOC2012_train_val"
#   Dataset evaluation


class UtilsConfigPath(PathConfig):

    def __init__(self):

        super(UtilsConfigPath, self).__init__()
        self.SAVE_MODEL = self.BASE_DIR / "save/ckpts"
        self.SAVE_FIGURES = self.BASE_DIR / "save/figures"
        self.SAVE_TEXT = self.BASE_DIR / "save/txt"
        self.PRETRAIN_MODEL = self.BASE_DIR / "pretrained"