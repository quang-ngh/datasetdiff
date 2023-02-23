import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class AbstractDataset(Dataset):
    def __init__(self, root: str, split: int, base_size, augment: bool, val: bool, scale: bool, *args, **kwargs) -> None:
        """
        --------------------------------
        Parameters:
            root: 
                root path of the dataset
            split: 
                ratio of split 
            base_size: 
                expected size of the dataset
            augment: 
                performing augmentation data if augment is true else do nothing
            val: 
                split the dataset into training set and validation set r.s.t the split if val is true, otherwise do nothing
            scale: 
                scale up or down the original image
            flip: 
                random flip the image
            augmentor: 
                combination of random transformations to augment the data
    """
        self.root = root
        self.splot = split
        self.base_size = base_size
        self.augment = augment
        self.val = val
        self.scale = scale
        self.toTensor = torchvision.transforms.ToTensor()  # Automatic scale images to [0,1]
        self.augmentor = None
        
        if self.augment:
            augment_composer = [
                torchvision.transforms.CenterCrop(self.base_size),
                torchvision.transforms.RandomHorizontalFlip(p = 0.5),
            ]
            self.augmentor = torchvision.transforms.Compose([
                *augment_composer
            ])
        super(AbstractDataset, self).__init__(**kwargs)
    
    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self):
        raise NotImplementedError
    

class ImageCaptioningDataset(AbstractDataset):

    def __init__(self, modal = "both") -> None:
        
        """
        Parameters:
            modal: ["image", "text"] 
                    dafault: fetch batches of pairs (image, text)
                    image: just fetch batches of images from the root 
                    text: just fetch batches of text input from the captions
                Examples:
                    dataset = ImageCaptioningDataset(root = ".", modal = "image")       # dataset just contains images
        """         
        super(ImageCaptioningDataset, self).__init__()
        self.modal = modal
        
        def _load_data(self, index: int):
            
            if not self.modal in ['image', 'text', 'both']:
                raise ValueError("Dataset must includes at least image or text! Domain is not compatible")
            
            if self.modal == "text":               # Just get the captions
                pass
            elif self.modal == "image":            # Just get the images
                pass
            else:                                  # Default: pairs of (image, text)
                pass
            
            pass
        
        def 
class SegmentationDataset(AbstractDataset):
    def __init__(self) -> None:
        super(SegmentationDataset, self).__init__()
        pass

