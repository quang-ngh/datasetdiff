import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader

def preprocess_get_indices(prompt: str):

    """
    Preprocess the prompt input and return the indices
    of singular noun

    prompt -> tokenizer -> stopwords -> filtered words
    """
    sent_tokens = word_tokenize(prompt)
    filtered_words = [word for word in sent_tokens if word not in stopwords.words('english')]
    pos_tags = pos_tag(filtered_words)
    
    indices = []
    for idx, item in enumerate(pos_tags):
        word, tag = item
        if tag == 'NN':
            indices.append(idx)
    
    return indices

def get_indices(prompt_list: tuple[str]):

    list_of_indices = [preprocess_get_indices(prompt) for prompt in prompt_list]
    return list_of_indices 

def prepare_model():
    pass

if __name__ == '__main__':

    image_caption_dataset = ImageCaptioningDataset(transform = None, train = "test")
    dataloader = DataLoader(image_caption_dataset, batch_size = 12)

    _, text = next(iter(dataloader))

    print(type(text))