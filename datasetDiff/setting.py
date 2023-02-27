from pathlib import Path
from enum import Enum

#   Path configuration
class PathConfig:
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_DIR = "data"
        self.DATA_IMG_CAPTIONING     = ["Flickr8K"]
        self.DATA_IMG_SEG            = ["Pascal-VOC-2012"]

        ################### Specific paths ##############
        self.CAPTIONS_PATH = self.BASE_DIR / self.DATA_DIR / self.DATA_IMG_CAPTIONING[0] / "captions"
        self.IMG_ID_PATH = self.BASE_DIR / self.DATA_DIR / self.DATA_IMG_CAPTIONING[0] / "images_id"
#   Dataset evaluation
