import torch
import torchvision
from torch.utils.data import Dataset
from setting import DataPathConfig
from glob import glob
import PIL
path = DataPathConfig()


class ImageCaptioningDataset(Dataset):

    def __init__(self, transform, **kwargs):
        
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
        
        self._set_files()
        self.images = None
        self.texts = None
        super(ImageCaptioningDataset, self).__init__()
        self._set_files()

        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((200, 200)),
                torchvision.transforms.ToTensor(),
            ])
    
    def get_text_files(self):

        file_path = path.CAPTIONS_PATH
        files = [filename for filename in file_path.glob("*.txt")]
        return files
    
    def get_images_idx_files(self):

        file_path = path.IMG_ID_PATH
        paths = file_path.glob("*.txt")
        img_paths = [img_path for img_path in paths]
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

class SegmentationDataset(Dataset):
    def __init__(self) -> None:
        super(SegmentationDataset, self).__init__()
        pass

