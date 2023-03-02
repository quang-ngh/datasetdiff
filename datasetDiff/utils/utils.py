import argparse
import os

def split_text_image(path, img_path, txt_path):
    file_path = os.path.join(os.getcwd(), path)
    imgs_path = os.path.join(os.getcwd(), img_path)
    txts_path = os.path.join(os.getcwd(), txt_path)
    
    string = None

    with open (path, 'r') as reader:
        
        string = reader.readlines()

    reader.close()
    print(string) 
    for idx, item in enumerate(string[1:]):

        img_id, caption = item.split('.jpg,')
        img_id += ".jpg"
        
        with open(imgs_path + "image_ids_"+ str(idx + 1) + ".txt", 'w') as writer:
            writer.write(img_id.strip("\n"))
        writer.close()

        with open(txt_path + "caption_"+str(idx + 1) + '.txt', 'w') as writer:
            writer.write(caption.strip("\n"))
    writer.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--img_path', type = str, default = "data/Flickr8K/images_id/")
    parser.add_argument('--txt_path', type=str, default= "data/Flickr8K/captions/")
    args = parser.parse_args()
    
    split_text_image(args.file_path, args.img_path, args.txt_path)
