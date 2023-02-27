import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from setting import PathConfig
from glob import glob
import PIL
path = PathConfig()


class ImageCaptioningDataset(Dataset):

    def __init__(self, **kwargs):
        
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
        
        if not kwargs["modal"] in ["text", "image", "both"]:

            if kwargs["modal"] is None:
                self.modal = "both"
            else:
                raise ValueError("Modal argument must be text, image or both. But found {}".format(kwargs['modal']))
        else:
            self.modal = kwargs['modal']
        self._set_files()
        self.images = None
        self.texts = None
        super(ImageCaptioningDataset, self).__init__(**kwargs)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = None
        text = None
        if self.modal == "both":
            image, text = self._load_data(index)
        elif self.modal == "image":
            image = self._load_data(index)
        elif self.modal == "text":
            text = self._load_data(index)
        else:
            raise ValueError("Modal must be one of the {}".format(["text", "image", "both"]))

        return image, text

    def get_text_files(self):

        file_path = path.CAPTIONS_PATH
        files = [file for file in file_path.glob(".*txt")]
        return files
    
    def get_images_idx_files(self):

        file_path = path.IMG_ID_PATH
        paths = file_path.glob(".*jpg") + file_path.glob("*.png") + file_path.glob("*.jepg")
        img_paths = [img_path for img_path in paths]
        return img_paths

    def read_img_file(self, filename):
        return self.toTensor(PIL.open(filename))
    
    def _set_files(self):
        
        #   Extract list of paths of files
        if self.modal == "text":
            self.files = self.get_text_files()

        #   Extract list of paths of files that contain the identity of 
        #   each image (r.sp.to text description)
        elif self.modal == "image":
            self.files = self.get_images_idx_files()
        
        elif self.modal == "both":
            txts = self.get_text_files()
            imgs = self.get_images_files()
            self.files = list(zip(txts, imgs))
            
    def _load_data(self, index: int):
        try: 
            if self.modal == "text":               
                """Return the string which is contained in the txt files.
                    Names of the text files are in the same order.
                """
                if not isinstance(self.files[index], str):
                    raise ValueError("Expect str type of content.")
                
                text  = ""
                fpath = self.files[index]
                with open(fpath, "r") as reader:
                    text = reader.read()
                reader.close()
                return text

            elif self.modal == "image":            
                ifpath = self.files[index]
                image = self.read_img_file(ifpath)
                return image
                            
            elif self.modal == "both":

                txt     =   None
                tfpath, ifpath = self.files[index]

                with open(tfpath, "r") as reader:
                    txt = reader.readlines()
                reader.close()
                txt     = txt.strip("\n")
                image   = self.read_img_file(ifpath)
                return image, txt  

            else: 
                raise ValueError("Modal must be one of {}".format(["text", "image", "both"]))
        except Exception as e:
            print(e)
        
class SegmentationDataset(AbstractDataset):
    def __init__(self) -> None:
        super(SegmentationDataset, self).__init__()
        pass
