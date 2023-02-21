import torch
from torch.utils.data import Dataset, DataLoader

class AbstractDataset(Dataset):
    def __init__(self, root: str, split: int, base_size, augment: bool, val: bool, scale: bool, flip: bool, *args, **kwargs) -> None:
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

        """
        self.root = root
        self.splot = split
        self.base_size = base_size
        self.augment = augment
        self.val = val
        self.scale = scale
        self.flip  = flip

        super(AbstractDataset, self).__init__(**kwargs)
    
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
        
        pass

class SegmentationDataset(AbstractDataset):
    def __init__(self) -> None:
        super(SegmentationDataset, self).__init__()
        pass

