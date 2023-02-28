from dataset import ImageCaptioningDataset, SegmentationDataset
from torch.utils.data import DataLoader
from setting import UtilsConfigPath
from tqdm import tqdm 

transform = None
flickr8k = ImageCaptioningDataset(transform, train = False)
flickr8kLoader = DataLoader(flickr8k, 4, False)

pascal_voc_2012 = SegmentationDataset(transform = None, train = False, setting = "test")
pascal_voc_loader = DataLoader(pascal_voc_2012, 12, False)


# def run_image_caption_VOC2012(captioner):
#     """
#         captioner: pre_trained image captioning model to extract caption
#                     from the image input
#         dataloader: data loader for the VOC
#     """

#     filename = "VOC2012Caption_"
#     image_captioner = captioner
#     dataloader = DataLoader(image_for_captioning)          # Just contains images of VOC2012 (train or train)

#     for idx, image in tqdm(enumerate(dataloader)):

#         caption = image_captioner(image)
#         with open(filename + str(idx+1) + '.txt', 'w') as writer:
#             writer.write(caption)
        
#         writer.close()
    

def run_attend_and_excite(pretrained, dataloader):

    filename  = "ATE_"

    for idx, batch in enumerate(dataloader):

        image, text = batch[0], batch[1]
        print(list(text))
        break

if __name__ == "__main__":   
    
    
    out = next(iter(pascal_voc_loader))
    print(out)
    #run_attend_and_excite(None, dataloader = flickr8kLoader)
