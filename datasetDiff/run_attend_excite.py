import nltk
import torch
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from dataset import ImageCaptioningDataset
from typing import List, Dict
from torch.utils.data import DataLoader
from attend_excite.pipeline_attend_and_excite import AttendAndExcitePipeline
from attend_excite.config import RunConfig
from attend_excite.run import run_on_prompt, get_indices_to_alter
from attend_excite.utils import vis_utils
from attend_excite.utils.ptp_utils import AttentionStore
from PIL import Image

def preprocess_get_indices(prompt: str):

    """
    Preprocess the prompt input and return the indices
    of singular noun

    prompt -> tokenizer -> stopwords -> filtered words
    """
    sent_tokens = word_tokenize(prompt)
    filtered_words = [word for word in sent_tokens if word not in stopwords.words('english')]
    pos_tags = pos_tag(filtered_words)

    print(pos_tags)  
    indices = []
    for idx, item in enumerate(pos_tags):
        word, tag = item
        if tag == 'NN':
            indices.append(idx)
    
    return indices

def get_indices(prompt_list: tuple[str]):

    list_of_indices = [preprocess_get_indices(prompt) for prompt in prompt_list]
    return list_of_indices

def run_and_display(prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    generator: torch.Generator,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
                    max_iter_to_alter: int = 25,
                    display_output: bool = False):
    config = RunConfig(prompt=prompts[0],
                       run_standard_sd=run_standard_sd,
                       scale_factor=scale_factor,
                       thresholds=thresholds,
                       max_iter_to_alter=max_iter_to_alter)
    image = run_on_prompt(model=stable,
                          prompt=prompts,
                          controller=controller,
                          token_indices=indices_to_alter,
                          seed=generator,
                          config=config)
    if display_output:
        display(image)
    return image

def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image

def get_model():
    ##  Get the model
    print("Load the model...")
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    stable = AttendAndExcitePipeline.from_pretrained("./stable-diffusion-v1-4", from_tf = True).to(device)
    tokenizer = stable.tokenizer
    print("Load model succesfully")
    return stable, tokenizer

def get_dataset():

    print("Load dataset...")
    image_caption_dataset = ImageCaptioningDataset(transform = None, train = "test", mode = "text")
    print("Get {}".format(image_caption_dataset.mode))
    dataloader = DataLoader(image_caption_dataset, batch_size = 2)
    text = next(iter(dataloader))
    print(text)
    sent_list_indices = get_indices(text)               # Get singular noun indices of each sentence in the batch
    assert len(text) == len(sent_list_indices), "Number of sentences must be equal to the number of indices list"
    print("Load dataset successfully")

    return text, sent_list_indices

def run(config: RunConfig):

    seeds = [123]
    stable_diff, tokenizer = get_model()
    ##  Running

    for seed in seeds:
    
        prompts = [string]
        controller = AttentionStore()
        token_indices = sent_list_indices[idx]
        print("Prompt: {}\nAlter indices = {}".format(prompts[0], token_indices))    
        print(f"Seed: {seed}")

        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                            model=stable_diff,
                            controller=controller,
                            token_indices=config.sent_list_indices,
                            seed=g,
                            config=config)
        # prompt_output_path = config.output_path / config.prompt
        # prompt_output_path.mkdir(exist_ok=True, parents=True)
        # image.save(prompt_output_path / f'{seed}.png')
        # images.append(image) 
        pseudo_semantic = vis_utils.show_cross_attention(
                                        prompt = config.prompt,
                                        attention_store = controller,
                                        tokenizer = tokenizer,
                                        indices_to_alter = token_indices,
                                        res = 16,
                                        from_where=("up", "down", "mid"),
                                        orig_image = image
                                    ) 
        
        print("Image shape = {}".format(image.shape))
            # vis_utils.show_cross_attention(attention_store=controller,
            #                                prompt=prompt,
            #                                tokenizer=tokenizer,
            #                                res=16,
            #                                from_where=("up", "down", "mid"),
            #                                indices_to_alter=token_indices,
            #                                orig_image=image)
if __name__ == '__main__':

    text, list_indices = get_dataset()
    if len(text) <= 0:
        raise ValueError("There is no prompt") 
    if len(text) == 1:
        config = RunConfig(text)
        config.sent_list_indices = list_indices[0]
        print("{}. Token indices = {}".format(config.prompt, config.sent_list_indices))
        run(config)
    else:
        for idx, prompt in enumerate(text):
            config = RunConfig(prompt)
            config.sent_list_indices = list_indices[idx]

            run(config) 
            del config

    # for idx, item in enumerate(sent_list_indices):
        # print("{} : {}".format(text[idx], sent_list_indices[idx]))
