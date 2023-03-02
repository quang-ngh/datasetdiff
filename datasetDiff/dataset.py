import torch
import torchvision
from torch.utils.data import Dataset
from setting import DataPathConfig
from glob import glob
import PIL
path = DataPathConfig()


class ImageCaptioningDataset(Dataset):

    def __init__(self, transform, train, **kwargs):
        
        """
        Parameters:
            root : the parent root of images, images_id, captions folders
            modal: ["image", "text"] 
                    dafault: fetch batches of pairs (image, text)
                    image: just fetch batches of images from the root 
                    text: just fetch batches of text input from the captions
                Examples:
                    dataset = ImageCaptioningDataset(root = ".", modal = "image")       # dataset just contains images
            
            images: list of image
        Methods:    
            get_text_files(self): 
                return list of *.txt files
            get_image_files(self):
                return list of *.png | *.jpg
            
            _set_files(self):
                assign file to be list (modal = (text,image)) or 
                dictionary (modal = both)
            
        """         
        
        self.images = None
        self.texts = None
        super(ImageCaptioningDataset, self).__init__()
        self._set_files()
        
        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((200, 200)),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        self.train = train
    def get_text_files(self):

        file_path = path.CAPTIONS_PATH 
        files = [filename for filename in file_path.glob("*.txt")]
        return files
    
    def get_images_idx_files(self):

        file_path = path.IMG_ID_PATH
        paths = file_path.glob("*.txt")
        img_paths = [img_path for img_path in paths]
        print(len(img_paths))
        return img_paths

    def read_img_file(self, filename):
        
        dst_part = None                       # Real image path
        with open(filename, "r") as f:
            dst_part = f.readlines()[0].strip("\n\r")
        f.close()
        fpath = path.IMG_CAPTION / dst_part
        return self.transform(PIL.Image.open(fpath))
    
    def _set_files(self):
        
        txts = self.get_text_files()
        imgs = self.get_images_idx_files()
        self.files = list(zip(txts, imgs))

    def _load_data(self, index: int):
        try:
            image   = None
            txt     =   None
            tfpath, ifpath = self.files[index]

            with open(tfpath, "r") as reader:
                txt = reader.readlines()
            reader.close()
            txt     = txt[0].strip("\n")
            image   = self.read_img_file(ifpath)

            return image, txt
        except Exception as e:
            print(e)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, txt = self._load_data(index)
        return image, txt
    
    def __str__(self):
        return str(path.IMG_CAPTION)


class SegmentationDataset(Dataset):
    def __init__(self, transform, train, *args, **kwargs) -> None:
        """
            transform : a compose of transformation that can be performed in
                        the torch tensor.
            setting:    semantic or instance segmentation
            train:      if train then load the train_val folder. Otherwise, load test folder
        """
        super(SegmentationDataset, self).__init__()
        
        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((200, 200)),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        if train:
            if  kwargs['setting'] == 'semantic':
                self.setting = 'semantic'
            elif kwargs['setting'] == 'instance':
                self.setting = 'instance'
            else:
                raise ValueError("Ambiguous setting. Segmentation should be semantic or instance. But found {}".format(kwargs['setting']))
        else:
            self.setting = None 

        self.semantic_masks = []                                    #   List of paths of semantic masks files
        self.images = []                                            #   List of paths of original images
        self.instance_masks = []                                    #   List of paths of instance segmentation masks files
        self.train = train
        self._set_files()

    def _set_files(self):


        
        if self.train :     
            jpeg_path  = path.VOC2012_TRAIN / "JPEGImages" 
            segment_classes = path.VOC2012_TRAIN / "SegmentationClasses"
            instance_classes = path.VOC2012_TRAIN / "SegmentationObject"

            self.images = [filepath for filepath in jpeg_path.glob("*.jpg")]
            
            if  self.setting == "segmentation":
                self.segment_classes = [filepath for filepath in segment_classes.glob("*.png")]
            elif  self.setting == "instance":
                self.instance_masks = [filepath for filepath in instance_classes.glob("*.png")]
            else:
                raise ValueError("Setting must be segmentation or ")

            if len(self.images) == 0 or len(self.segment_classes) == 0 or len(self.instance_masks):
                raise ValueError("One of three dataset is empty (Images, Semantic, Instance)")
        else:
            jpeg_path = path.VOC2012_TEST_IMG
            print(jpeg_path)
            self.images = [filepath for filepath in jpeg_path.glob("*.jpg")]
            if len(self.images) == 0:
                raise ValueError ("There are no train images. Emtpy images list")


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        if self.train:
            image       = None
            semantic    = None
            instance    = None
            
            try:
                image = self.transform(PIL.Image.open(self.images[index]))
                semantic = self.transform(PIL.Image.open(self.semantic_masks[index]))
                instance = self.transform(PIL.Image.open(self.instance_masks[index]))
            except Exception as e:
                print(e)
                raise ValueError("Cannot open image or semantic or instace")

            return image, semantic, instance
        else:
            image   = self.transform(PIL.Image.open(self.images[index]))
            return image


        
