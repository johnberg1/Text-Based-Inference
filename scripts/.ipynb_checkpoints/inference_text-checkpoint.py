from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
import sys
sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im
from models.psp import pSp # takes around 1 min

EXPERIMENT_TYPE = 'celeba_encode'

EXPERIMENT_DATA_ARGS = {
    "celeba_encode": {
        "model_path": "../pretrained_models/best_model_10.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celeba_encode_25": {
        "model_path": "../pretrained_models/best_model_25.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celeba_encode_15": {
        "model_path": "exp_hyper_1/checkpoints/best_model.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celeba_encode_20": {
        "model_path": "exp_hyper_2/checkpoints/best_model.pt",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS_15 = EXPERIMENT_DATA_ARGS["celeba_encode_15"]
EXPERIMENT_ARGS_20 = EXPERIMENT_DATA_ARGS["celeba_encode_20"]


model_path_15 = EXPERIMENT_ARGS_15['model_path']
ckpt_15 = torch.load(model_path_15, map_location='cpu')
model_path_20 = EXPERIMENT_ARGS_20['model_path']
ckpt_20 = torch.load(model_path_20, map_location='cpu')

opts_15 = ckpt_15['opts']
opts_20 = ckpt_20['opts']

opts_15['checkpoint_path'] = model_path_15
opts_20['checkpoint_path'] = model_path_20

opts_15 = Namespace(**opts_15)
net_15 = pSp(opts_15)
net_15.eval()
net_15.cuda()

opts_20 = Namespace(**opts_20)
net_20 = pSp(opts_20)
net_20.eval()
net_20.cuda()

image_path = "/scratch/users/abaykal20/e4e/encoder4editing/train_images/15116.jpg"
original_image = Image.open(image_path).convert("RGB")
original_image.resize((256, 256))

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')

custom_caption = "This man has black hair and beard. He is wearing necktie."
img_transforms = EXPERIMENT_ARGS_15['transform']
input_image = img_transforms(original_image)
input_image = input_image.unsqueeze(0)

text_input = clip.tokenize(custom_caption)

text_input = text_input.cuda()
with torch.no_grad():
    text_features = clip_model.encode_text(text_input).float()
    
results = np.array(original_image) #np.array(aligned_image.resize((1024, 1024)))
print(f"Running on custom caption: {custom_caption}")
with torch.no_grad():
  result_tensor_15 = net_15(input_image.cuda().float(), text_features, randomize_noise=False, resize=False)
  result_tensor_15 = result_tensor_15.squeeze(0)
  result_image_15 = tensor2im(result_tensor_15)

  result_tensor_20 = net_20(input_image.cuda().float(), text_features, randomize_noise=False, resize=False)
  result_tensor_20 = result_tensor_20.squeeze(0)
  result_image_20 = tensor2im(result_tensor_20)

  #results = np.concatenate([results, result_image_15, result_image_20], axis=1)

result_image_15.save("tedi_res/custom_res15.jpg")
result_image_20.save("tedi_res/custom_res20.jpg")
