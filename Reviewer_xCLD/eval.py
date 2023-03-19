import preclip.clip as clip
import torch
from torchvision import transforms
import argparse
from torch.utils.data import Dataset
from diffusers import StableDiffusionPipeline,DDIMScheduler,DPMSolverMultistepScheduler
from pathlib import Path
import numpy as np
import hashlib
from tqdm.auto import tqdm
import re
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        images_dir,
        size=512,
        hflip=False
    ):
        self.size = size


        self.instance_images_path = []


     
        inst_img_path = [x for x in Path(images_dir).iterdir() if x.is_file()]
        self.instance_images_path.extend(inst_img_path)

        
        self.num_instance_images = len(self.instance_images_path)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize([size,size] ,interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        return example

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where to save image with all prompts",
    )
    parser.add_argument(
        '--prompts', 
        action='append',
        help='get several prompts as : --prompts prompts0 --prompts prompts1--prompts prompts2',
        type=str
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="text-inversion-model",
        help="The directory where to save your own model"
    )
    parser.add_argument(
        "--src_images",
        action='append',
        type=str,
        help="The src images dir"
    )
    args = parser.parse_args()
    return args
def load_model(text_encoder, tokenizer,save_path):
    
    print("loading embeddings")
    
    st = torch.load(save_path)
    if 'modifier_token' in st:
        modifier_tokens = list(st['modifier_token'].keys())
        print(modifier_tokens)
        modifier_token_id = []
        for modifier_token in modifier_tokens:
            _ = tokenizer.add_tokens(modifier_token)
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, id_ in enumerate(modifier_token_id):
            token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]
class CLIPEvaluator(object):
    def __init__(self, device=torch.device("cuda"), clip_model='ViT-L/14'):
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images):
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text, norm = True):

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img, norm = True):
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()
class  PromptDataset(Dataset):

    def __init__(self, prompt, num_samples=50):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
def get_images_output(prompt,class_images_dir,pipeline):
    
    cur_class_images = len(list(class_images_dir.iterdir()))
    if cur_class_images>49:
        return None
    sample_dataset = PromptDataset(prompt)
    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)
    for example in tqdm(sample_dataloader, desc="Generating class images", ):
        images = pipeline(example["prompt"],num_inference_steps=50,guidance_scale=7.5).images
        for i, image in enumerate(images):
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}.{prompt}.png"
            if (example['index'][i] + cur_class_images)> 49:
                return None
            image.save(image_filename)
    return 'have done'


def multiple_replace(text, idict):  
    rx = re.compile('|'.join(map(re.escape, idict)))  
    def one_xlat(match):  
        return idict[match.group(0)]  
    return rx.sub(one_xlat, text) 

def main(args):
    
    pipe = StableDiffusionPipeline.from_pretrained("", torch_dtype=torch.float32, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    load_model(pipe.text_encoder, pipe.tokenizer,args.model_path+"/delta.bin")
    mask_dict=np.load(args.model_path+'/mask.npy',allow_pickle=True)
    for i in pipe.unet.named_parameters():
        if 'to_v' in i[0] or'to_k' in i[0]:
            i[1].data= torch.from_numpy(mask_dict.item()[i[0][:-6]+'mask_real']).float().cuda()*i[1].data
    sim_img=[[] for i in range(len(args.src_images))]
    print(sim_img)
    sim_text=[]
    evaluator=CLIPEvaluator(device=torch.device("cuda"), clip_model='ViT-B/32')
    idict={'ggd':"", 'kfn':"",'sks':""}
    for prompt in args.prompts:
        class_images_dir = Path(args.image_dir+'/'+prompt)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        state=get_images_output(prompt,class_images_dir,pipe)
        images_prompt_loader= DreamBoothDataset(class_images_dir, size=512)
        images_prompt = [images_prompt_loader[i]["instance_images"] for i in range(images_prompt_loader.num_instance_images)]
        generated_images = torch.stack(images_prompt, axis=0)
        for i in range(len(args.src_images)):
            data_loader = DreamBoothDataset(args.src_images[i], size=512)
            images = [data_loader[i]["instance_images"] for i in range(data_loader.num_instance_images)]
            images = torch.stack(images, axis=0)
            sim_samples_to_img  = evaluator.img_to_img_similarity(images, generated_images)
            sim_img[i].append(sim_samples_to_img.item())
        print(prompt.replace("ggd ",'').replace("kfn ",'').replace("sks ",''))
        sim_samples_to_text = evaluator.txt_to_img_similarity(prompt.replace("ggd ",'').replace("kfn ",'').replace("sks ",''),  generated_images)
        sim_text.append(sim_samples_to_text.item())
    mean_of_all=0
    for i in range(len(args.src_images)):
        print(f"img_alignment is equal to_{sim_img[i]}_target_{i}_mean{sum(sim_img[i])/len(args.prompts)}\n")
        mean_of_all+=(sum(sim_img[i])/len(args.prompts))
    print(f'mean of all is equal to _{mean_of_all/len(args.src_images)}\n')
    print(f"text_alignment is equal to_{sim_text}_mean{sum(sim_text)/len(args.prompts)}")
    print(args.model_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)



