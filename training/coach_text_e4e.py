import os
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm
from configs import data_configs
# from datasets.images_dataset import ImagesDataset
from datasets.images_text_dataset import ImagesTextDataset
# from datasets.augmentations import AgeTransformer
from criteria.lpips.lpips import LPIPS
# from criteria.aging_loss import AgingLoss
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.e4e_encoders_adain import ProgressiveStage
from criteria.clip_loss import CLIPLoss, DirectionalCLIPLoss
from models.e4e import pSp
from training.ranger import Ranger
import clip


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda'
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)
		self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device = self.device)
		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(opts=self.opts)
		self.clip_loss = CLIPLoss(self.clip_model)
		self.directional_loss = DirectionalCLIPLoss(self.clip_model)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

        # Initialize discriminator
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
			self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
			self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
			self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)
            
		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)


		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def perform_forward_pass(self, x, txt_embed):
		y_hat, latent = self.net.forward(x, txt_embed, return_latents=True)
		return y_hat, latent

	def __set_target_to_source(self, x, input_ages):
		return [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to(self.device)))
				for img, age in zip(x, input_ages)]
    
	def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
		for i in range(len(self.opts.progressive_steps)):
			if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))
			if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))

# Add opts.progressive_steps, w_discriminator_lambda, use_w_pool, d_reg_every, delta_norm_lambda
	def train(self):
		self.net.train()
		if self.opts.progressive_steps:
			self.check_for_progressive_training_update()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, y, txt = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				text_original = clip.tokenize(txt).to(self.device)
				with torch.no_grad():
					txt_embed_original = self.clip_model.encode_text(text_original)
				txt_embed_original = txt_embed_original.to(self.device).float()
				self.optimizer.zero_grad()
        
				mismatch_text = random.random() <= (3. / 4)
				if mismatch_text:
					txt_embed_mismatch = torch.roll(txt_embed_original, 1, dims=0)
					text_mismatch = torch.roll(text_original, 1, dims=0)
				else:
					txt_embed_mismatch = txt_embed_original
					text_mismatch = text_original
                    
				loss_dict_disc = {}
				if self.is_training_discriminator():
					loss_dict_disc = self.train_discriminator(x, txt_embed_mismatch, data_type="real")
                    
				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x, txt_embed_mismatch)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, text_original, text_mismatch, latent, y, mismatch_text, data_type="real")
				loss_dict = {**loss_dict, **loss_dict_disc}
				loss.backward()

				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				txt_embed_clone = txt_embed_original.clone().detach().requires_grad_(True)
				text_clone = text_original
                    
				cycle_loss_dict_disc = {}
				if self.is_training_discriminator():
					cycle_loss_dict_disc = self.train_discriminator(y_hat_clone, txt_embed_clone, data_type="cycle")
                    
        
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_clone, txt_embed_clone)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, text_mismatch, text_clone, latent_cycle, y_hat_clone, mismatch_text, data_type="cycle")
				cycle_loss_dict = {**cycle_loss_dict, **cycle_loss_dict_disc}
                
				loss.backward()
				self.optimizer.step()

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				loss_dict.update(cycle_loss_dict)
				loss_dict["loss"] = loss_dict["loss_real"] + loss_dict["loss_cycle"]

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or \
						(self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, txt, mismatch_text,
											  title='images/train/faces')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
				if self.opts.progressive_steps:
					self.check_for_progressive_training_update()
                
	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y, txt = batch
			text_original = clip.tokenize(txt).to(self.device)
			with torch.no_grad():
				txt_embed_original = self.clip_model.encode_text(text_original)
				txt_embed_original = txt_embed_original.to(self.device).float()
				x, y = x.to(self.device).float(), y.to(self.device).float()

				

				mismatch_text = random.random() <= (3. / 4)
				if mismatch_text:
					txt_embed_mismatch = torch.roll(txt_embed_original, 1, dims=0)
					text_mismatch = torch.roll(text_original, 1, dims=0)
				else:
					txt_embed_mismatch = txt_embed_original
					text_mismatch = text_original
                    
				cur_loss_dict_disc = {}
				if self.is_training_discriminator():
					cur_loss_dict_disc = self.validate_discriminator(x, txt_embed_mismatch, data_type="real")
            
				# perform forward/backward pass on real images
				y_hat, latent = self.perform_forward_pass(x, txt_embed_mismatch)
				_, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, text_original, text_mismatch, latent, y, mismatch_text, data_type="real")
				cur_loss_dict = {**cur_loss_dict, **cur_loss_dict_disc}

				# perform cycle on generate images by setting the target ages to the original input ages
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				txt_embed_clone = txt_embed_original.clone().detach().requires_grad_(True)
				text_clone = text_original
                
				cycle_loss_dict_disc = {}
				if self.is_training_discriminator():
					cycle_loss_dict_disc = self.validate_discriminator(y_hat_clone, txt_embed_clone, data_type="cycle")
                
				y_recovered, latent_cycle = self.perform_forward_pass(y_hat_clone, txt_embed_clone)
				loss, cycle_loss_dict, cycle_id_logs = self.calc_loss(x, y, y_recovered, text_mismatch, text_clone, latent_cycle, y_hat_clone, mismatch_text, data_type="cycle")
				cycle_loss_dict = {**cycle_loss_dict, **cycle_loss_dict_disc}

				# combine the logs of both forwards
				for idx, cycle_log in enumerate(cycle_id_logs):
					id_logs[idx].update(cycle_log)
				cur_loss_dict.update(cycle_loss_dict)
				cur_loss_dict["loss"] = cur_loss_dict["loss_real"] + cur_loss_dict["loss_cycle"]

			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat, y_recovered, txt, mismatch_text, title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, '
						'Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesTextDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts, train=True)
		test_dataset = ImagesTextDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts, train=False)
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, source_text, target_text, latent, directional_source, mismatch_text, data_type="real"):
		loss_dict = {}
		id_logs = []
		loss = 0.0
                
		if self.is_training_discriminator(): # Adversarial loss
			loss_disc = 0.
			dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))
                
			for i in dims_to_discriminate:
				w = latent[:, i, :]
				fake_pred = self.discriminator(w)
				loss_disc += F.softplus(-fake_pred).mean()
			loss_disc /= len(dims_to_discriminate)
			loss_dict[f'encoder_discriminator_loss_{data_type}'] = float(loss_disc)
			loss += self.opts.w_discriminator_lambda * loss_disc
            
		if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 18:  # delta regularization loss
			total_delta_loss = 0
			deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()
            
			first_w = latent[:, 0, :]
			for i in range(1, self.net.encoder.progressive_stage.value + 1):
				curr_dim = deltas_latent_dims[i]
				delta = latent[:, curr_dim, :] - first_w
				delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
				loss_dict[f"delta{i}_loss_{data_type}"] = float(delta_loss)
				total_delta_loss += delta_loss
			loss_dict['total_delta_loss_{data_type}'] = float(total_delta_loss)
			loss += self.opts.delta_norm_lambda * total_delta_loss
            
		if self.opts.id_lambda > 0:
			weights = None
			if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
				age_diffs = torch.abs(target_ages - input_ages)
				weights = train_utils.compute_cosine_weights(x=age_diffs)
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, label=data_type, weights=weights)
			loss_dict[f'loss_id_{data_type}'] = float(loss_id)
			loss_dict[f'id_improve_{data_type}'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'loss_l2_{data_type}'] = float(loss_l2)
			if data_type == "real":
				l2_lambda = self.opts.l2_lambda_aging
			else:
				l2_lambda = self.opts.l2_lambda
			loss += loss_l2 * l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'loss_lpips_{data_type}'] = float(loss_lpips)
			if data_type == "real":
				lpips_lambda = self.opts.lpips_lambda_aging
			else:
				lpips_lambda = self.opts.lpips_lambda
			loss += loss_lpips * lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, latent_avg=self.net.latent_avg)
			loss_dict[f'loss_w_norm_{data_type}'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		# CLIP loss
		#loss_clip = self.clip_loss(y_hat, target_text).diag().mean()
		#loss_dict[f'loss_clip_{data_type}'] = float(loss_clip)
		#loss += loss_clip * 1.0
		if mismatch_text: 
			loss_directional = self.directional_loss(directional_source, y_hat, source_text, target_text).mean()
			loss_dict[f'loss_directional_{data_type}'] = float(loss_directional)
			loss += loss_directional * 1.0
   
		loss_dict[f'loss_{data_type}'] = float(loss)
		if data_type == "cycle":
			loss = loss * self.opts.cycle_lambda
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, y_recovered, txt, mismatch_text, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
				'recovered_face': common.tensor2im(y_recovered[i])
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, txt, mismatch_text, im_data=im_data, subscript=subscript)

	def log_images(self, name, txt, mismatch_text, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data, txt, mismatch_text)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.net.latent_avg is not None:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict

	def get_dims_to_discriminate(self):
		deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
		return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]
    
	def is_progressive_training(self):
		return self.opts.progressive_steps is not None
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def is_training_discriminator(self):
		return self.opts.w_discriminator_lambda > 0
    
    
	@staticmethod
	def discriminator_loss(real_pred, fake_pred, loss_dict):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()
    
		loss_dict['d_real_loss'] = float(real_loss)
		loss_dict['d_fake_loss'] = float(fake_loss)
    
		return real_loss + fake_loss
    
	@staticmethod
	def discriminator_r1_loss(real_pred, real_w):
		grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    
		return grad_penalty

	@staticmethod
	def requires_grad(model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag
            

	def train_discriminator(self, x, txt_embed, data_type="real"):
		loss_dict = {}
		self.requires_grad(self.discriminator, True)
		with torch.no_grad():
			real_w, fake_w = self.sample_real_and_fake_latents(x, txt_embed)
		real_pred = self.discriminator(real_w)
		fake_pred = self.discriminator(fake_w)
		loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
		loss_dict[f'discriminator_loss_{data_type}'] = float(loss)
            
		self.discriminator_optimizer.zero_grad()
		loss.backward()
		self.discriminator_optimizer.step()
            
		# r1 regularization
		d_regularize = self.global_step % self.opts.d_reg_every == 0
		if d_regularize:
			real_w = real_w.detach()
			real_w.requires_grad = True
			real_pred = self.discriminator(real_w)
			r1_loss = self.discriminator_r1_loss(real_pred, real_w)
            
			self.discriminator.zero_grad()
			r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
			r1_final_loss.backward()
			self.discriminator_optimizer.step()
			loss_dict[f'discriminator_r1_loss_{data_type}'] = float(r1_final_loss)
            
		# Reset to previous state
		self.requires_grad(self.discriminator, False)
        
		return loss_dict
    
	def validate_discriminator(self, x, txt_embed, data_type="real"):
		with torch.no_grad():
			loss_dict = {}
			real_w, fake_w = self.sample_real_and_fake_latents(x, txt_embed)
			real_pred = self.discriminator(real_w)
			fake_pred = self.discriminator(fake_w)
			loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
			loss_dict[f'discriminator_loss_{data_type}'] = float(loss)
			return loss_dict
        
	def sample_real_and_fake_latents(self, x, txt_embed):
		sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
		real_w = self.net.decoder.get_latent(sample_z)
		fake_w = self.net.encoder(x, txt_embed)
		if self.opts.start_from_latent_avg:
			fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
		if self.opts.start_from_encoded_w_plus:
			with torch.no_grad():
				encoded_latents = self.net.pretrained_encoder(x)
				encoded_latents = encoded_latents + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
			fake_w = fake_w + encoded_latents
		if self.is_progressive_training():  # When progressive training, feed only unique w's
			dims_to_discriminate = self.get_dims_to_discriminate()
			fake_w = fake_w[:, dims_to_discriminate, :]
		if self.opts.use_w_pool:
			real_w = self.real_w_pool.query(real_w)
			fake_w = self.fake_w_pool.query(fake_w)
		if fake_w.ndim == 3:
			fake_w = fake_w[:, 0, :]
		return real_w, fake_w