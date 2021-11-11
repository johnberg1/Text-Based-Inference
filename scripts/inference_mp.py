from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.images_text_dataset import ImagesTextDataset
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp
import clip

def run():
	test_opts = TestOptions().parse()

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()
	
	clip_model, clip_preprocess = clip.load("ViT-B/32", device = 'cuda')
	upsample = nn.Upsample(scale_factor=7)
	avg_pool = nn.AvgPool2d(kernel_size=256 // 32)
	
	print(f'Loading dataset for {opts.dataset_type}')
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	
	test_dataset = ImagesTextDataset(source_root=dataset_args['test_source_root'],
							 target_root=dataset_args['test_target_root'],
							 source_transform=transforms_dict['transform_inference'],
							 target_transform=transforms_dict['transform_inference'],
							 opts=opts, train=False)
	dataloader = DataLoader(test_dataset,
			  batch_size=opts.test_batch_size,
			  shuffle=False,
			  num_workers=int(opts.test_workers),
			  drop_last=False)


	
	if opts.n_images is None:
		opts.n_images = len(test_dataset)
		print("Number of images", opts.n_images)
	
	global_time = []	
	means = []
	for batch_idx, batch in enumerate(dataloader):
		x,_,txt = batch
		text_original = clip.tokenize(txt).to('cuda')
		with torch.no_grad():
			txt_embed_original = clip_model.encode_text(text_original)
			txt_embed_original = txt_embed_original.to('cuda').float()
			x= x.to('cuda').float()
			
			txt_embed_mismatch = torch.roll(txt_embed_original, 1, dims=0)
			text_mismatch = torch.roll(text_original, 1, dims=0)
			tic = time.time()
			y_hat = net.forward(x, txt_embed_mismatch)
			print(torch.min(x[1,:,:,:]))
			print(torch.max(x[1,:,:,:]))
			print(y_hat.shape)
			print(y_hat[1,:,:,:])
			print(torch.norm(y_hat[1,:,:,:],p=1))
			print(torch.min(y_hat[1,:,:,:]))
			print(torch.max(y_hat[1,:,:,:]))
			sys.exit("here")
			toc = time.time()
			global_time.append(toc - tic)
			y_hat_clip = avg_pool(upsample(y_hat))
			clip_similarity = clip_model(y_hat_clip,text_mismatch)[0] / 100
			means.append((clip_similarity.diag() * (1 - torch.norm(x - y_hat,p=1,dim=(1,2,3))/(x.shape[1] * x.shape[2] * x.shape[3]))).mean().item())
			
	stats_path = os.path.join(opts.exp_dir,'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)
	avg_sim = sum(means) / len(means)
	print("Manipulative Precision", avg_sim)
	

	with open(stats_path, 'w') as f:
		f.write(result_str)
		f.write("Manipulative Precision is {}".format(avg_sim))


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


if __name__ == '__main__':
	run()
